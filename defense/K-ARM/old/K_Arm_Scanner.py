import torch 
import numpy as np 
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm



class K_Arm_Scanner:
    def __init__ (self,model,args):
        self.model = model
        self.regularization = args.regularization
        self.init_cost = [args.init_cost] * args.num_classes
        self.steps = args.step
        self.round = args.rounds
        self.lr = args.lr
        self.num_classes = args.num_classes
        self.attack_succ_threshold = args.attack_succ_threshold
        self.patience = args.patience
        self.channels = args.channels
        self.batch_size = args.batch_size
        self.mask_size = [1,args.input_width,args.input_height]

        self.single_color_opt = args.single_color_opt

        self.pattern_size = [1,args.channels,args.input_width,args.input_height]

        
        #K-arms bandits
        self.device = torch.device("cuda:%d" % args.device)
        self.epsilon = args.epsilon
        self.epsilon_for_bandits = args.epsilon_for_bandits
        self.beta = args.beta
        self.warmup_rounds = args.warmup_rounds

        self.cost_multiplier = args.cost_multiplier 
        self.cost_multiplier_up = args.cost_multiplier
        self.cost_multiplier_down = args.cost_multiplier

        self.early_stop = args.early_stop
        self.early_stop_threshold = args.early_stop_threshold
        self.early_stop_patience = args.early_stop_patience
        self.reset_cost_to_zero = args.reset_cost_to_zero


        self.mask_tanh_tensor = [torch.zeros(self.mask_size).to(self.device)] * self.num_classes 

        self.pattern_tanh_tensor = [torch.zeros(self.pattern_size).to(self.device)] * self.num_classes

        self.pattern_raw_tensor = []
        self.mask_tensor = []
        for i in range(self.num_classes):
            self.pattern_raw_tensor.append(torch.tanh(self.pattern_tanh_tensor[i]) / (2 - self.epsilon) + 0.5)
            self.mask_tensor.append(torch.tanh(self.mask_tanh_tensor[i]) / (2 - self.epsilon) + 0.5)


        
    def reset_state(self,pattern_init,mask_init):
        if self.reset_cost_to_zero:
            self.cost = [0] * self.num_classes
        else:
            self.cost = self.init_cost
        self.cost_tensor = self.cost


        mask_np = mask_init.cpu().numpy()
        mask_tanh = np.arctanh((mask_np - 0.5) * (2-self.epsilon))
        mask_tanh = torch.from_numpy(mask_tanh).to(self.device)

        pattern_np = pattern_init.cpu().numpy()
        pattern_tanh = np.arctanh((pattern_np - 0.5) * (2 - self.epsilon))
        pattern_tanh = torch.from_numpy(pattern_tanh).to(self.device)

        for i in range(self.num_classes):
            self.mask_tanh_tensor[i] = mask_tanh.clone()
            self.pattern_tanh_tensor[i] = pattern_tanh.clone()
            self.mask_tanh_tensor[i].requires_grad = True
            self.pattern_tanh_tensor[i].requires_grad = True


    def update_tensor(self,mask_tanh_tensor,pattern_tanh_tensor,y_target_index,first=False):


        if first is True:
            for i in range(self.num_classes):
                self.mask_tensor[i] = (torch.tanh(mask_tanh_tensor[i]) / (2 - self.epsilon) + 0.5)
                self.pattern_raw_tensor[i] = (torch.tanh(pattern_tanh_tensor[i]) / ( 2 - self.epsilon) + 0.5)
        
        else:

            self.mask_tensor[y_target_index] = (torch.tanh(mask_tanh_tensor[y_target_index]) / (2 - self.epsilon) + 0.5)
            self.pattern_raw_tensor[y_target_index] = (torch.tanh(pattern_tanh_tensor[y_target_index]) / (2 - self.epsilon) + 0.5)


    def scanning(self,target_classes_all,data_loader_arr,y_target_index,pattern_init,mask_init,trigger_type,direction):
        self.reset_state(pattern_init,mask_init)
        # for TrojAI round1, the data format is BGR, then need permute
        #permute = [2,1,0]

        self.update_tensor(self.mask_tanh_tensor,self.pattern_tanh_tensor,y_target_index,True)


        #K-arms bandits version
        best_mask = [None] * self.num_classes
        best_pattern = [None] * self.num_classes
        best_reg = [1e+10] * self.num_classes

        best_acc = [0] * self.num_classes



        log = []
        cost_set_counter = [0] * self.num_classes
        cost_down_counter = [0] * self.num_classes
        cost_up_counter = [0] * self.num_classes
        cost_up_flag = [False] * self.num_classes
        cost_down_flag = [False] * self.num_classes
        early_stop_counter = [0] * self.num_classes
        early_stop_reg_best = [1e+10] * self.num_classes
        early_stop_tag = [False] * self.num_classes
        update = [False] * self.num_classes


        avg_loss_ce = [1e+10] * self.num_classes
        avg_loss_reg = [1e+10] * self.num_classes
        avg_loss = [1e+10] * self.num_classes
        avg_loss_acc = [1e+10] * self.num_classes
        reg_down_vel = [-1e+10] * self.num_classes
        times = [0] * self.num_classes
        total_times = [0] * self.num_classes
        first_best_reg = [1e+10] * self.num_classes
        y_target_tensor = torch.Tensor([target_classes_all[y_target_index]]).long().to(self.device)
        
        optimizer_list = []


        
        for i in range(self.num_classes):
            optimizer = optim.Adam([self.pattern_tanh_tensor[i],self.mask_tanh_tensor[i]],lr=self.lr,betas=(0.5,0.9))
            optimizer_list.append(optimizer)

        criterion = nn.CrossEntropyLoss()

        pbar = tqdm(range(self.steps))

        #for step in range(self.steps):
        
        for step in pbar:

            y_target_tensor = torch.Tensor([target_classes_all[y_target_index]]).long().to(self.device)
            total_times[y_target_index] += 1


            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []
            for idx, (img,name,label) in enumerate(data_loader_arr[y_target_index]):
                img = img.to(self.device)
                Y_target = y_target_tensor.repeat(img.size()[0])

                X_adv_tensor = (1-self.mask_tensor[y_target_index]) * img + self.mask_tensor[y_target_index] * self.pattern_raw_tensor[y_target_index]


                optimizer_list[y_target_index].zero_grad()

                output_tensor = self.model(X_adv_tensor)

                pred = output_tensor.argmax(dim=1, keepdim=True)


                self.loss_acc = pred.eq(Y_target.long().view_as(pred)).sum().item() / (img.size()[0])
                self.loss_ce = criterion(output_tensor,Y_target)

                self.loss_reg = torch.sum(torch.abs(self.mask_tensor[y_target_index] ))


                self.loss = self.loss_ce + self.loss_reg * self.cost_tensor[y_target_index] 


                self.loss.backward()

                optimizer_list[y_target_index].step()
                self.update_tensor(self.mask_tanh_tensor,self.pattern_tanh_tensor,y_target_index)


                pbar.set_description('Target: {}, victim: {}, Loss: {:.4f}, Acc: {:.2f}%, CE_Loss: {:.2f}, Reg_Loss:{:.2f}, Cost:{:.2f} best_reg:{:.2f} avg_loss_reg:{:.2f}'.format(
                target_classes_all[y_target_index], label[0], self.loss,self.loss_acc * 100,self.loss_ce,self.loss_reg,self.cost_tensor[y_target_index],best_reg[y_target_index],avg_loss_reg[y_target_index]))
                loss_ce_list.append(self.loss_ce.item())
                loss_reg_list.append(self.loss_reg.item())
                loss_list.append(self.loss.item())
                loss_acc_list.append(self.loss_acc)

            
            #K-arms Bandits
            avg_loss_ce[y_target_index] = np.mean(loss_ce_list)
            avg_loss_reg[y_target_index]= np.mean(loss_reg_list)
            avg_loss[y_target_index] = np.mean(loss_list)
            avg_loss_acc[y_target_index] = np.mean(loss_acc_list)
            
            if avg_loss_acc[y_target_index] > best_acc[y_target_index]:
                best_acc[y_target_index] = avg_loss_acc[y_target_index]
            
            
            if direction == 'forward':
        
                if (total_times[y_target_index] > 20 and best_acc[y_target_index] < 0.3 and trigger_type == 'polygon_specific') or (total_times[y_target_index] > 200 and  best_acc[y_target_index] < 0.8 and trigger_type == 'polygon_specific') or (total_times[y_target_index] > 10 and  best_acc[y_target_index] == 0 and trigger_type == 'polygon_specific'):
            
                    early_stop_tag[y_target_index] = True
            
            elif direction == 'backward':
                if (total_times[y_target_index] > 200 and  best_acc[y_target_index] < 1 and trigger_type == 'polygon_specific'):
                    #for the backward check
                    early_stop_tag[y_target_index] = True

            update[y_target_index] = False
            if avg_loss_acc[y_target_index] >= self.attack_succ_threshold and avg_loss_reg[y_target_index] < best_reg[y_target_index]:
                best_mask[y_target_index] = self.mask_tensor[y_target_index]



                #print('best_mask update')
                update[y_target_index] = True
                times[y_target_index] += 1
                best_pattern[y_target_index] = self.pattern_raw_tensor[y_target_index]
                
                if times[y_target_index] == 1:
                    first_best_reg[y_target_index] = 2500
                    #self.cost_tensor[y_target_index] = 1e-3
                #reg_down_vel[y_target_index] = 1e+4 * (np.log10(first_best_reg[y_target_index]) - np.log10(avg_loss_reg[y_target_index])) / (times[y_target_index] + total_times[y_target_index] / 10)
                reg_down_vel[y_target_index] =  ((first_best_reg[y_target_index]) - (avg_loss_reg[y_target_index])) / (times[y_target_index] + (total_times[y_target_index] / 2))
                #print('best_reg:',best_reg[y_target_index])
                #print('avg_loss_reg:',avg_loss_reg[y_target_index])

                best_reg[y_target_index] = avg_loss_reg[y_target_index]

            if self.early_stop:

                if best_reg[y_target_index] < 1e+10:
                    if best_reg[y_target_index] >= self.early_stop_threshold * early_stop_reg_best[y_target_index]:
                        #print('best_reg:',best_reg[y_target_index])
                        #print('early_stop_best_reg:',early_stop_reg_best[y_target_index])
                        early_stop_counter[y_target_index] +=1
                    else:
                        early_stop_counter[y_target_index] = 0
                early_stop_reg_best[y_target_index] = min(best_reg[y_target_index],early_stop_reg_best[y_target_index])

                if (times[y_target_index] > self.round) or (cost_down_flag[y_target_index] and cost_up_flag[y_target_index] and  early_stop_counter[y_target_index] > self.early_stop_patience and trigger_type == 'polygon_global'):

                    if y_target_index  == torch.argmin(torch.Tensor(best_reg)):
                        print('early stop for all!')
                        break
                    else:
                        early_stop_tag[y_target_index] = True

                        if all(ele == True for ele in early_stop_tag):
                            break


            if early_stop_tag[y_target_index] == False:

                if self.cost[y_target_index] == 0 and avg_loss_acc[y_target_index] >= self.attack_succ_threshold:
                    cost_set_counter[y_target_index] += 1
                    if cost_set_counter[y_target_index] >= 2:
                        self.cost[y_target_index] = self.init_cost[y_target_index]
                        self.cost_tensor[y_target_index] =  self.cost[y_target_index]
                        cost_up_counter[y_target_index] = 0
                        cost_down_counter[y_target_index] = 0
                        cost_up_flag[y_target_index] = False
                        cost_down_flag[y_target_index] = False
                else:
                    cost_set_counter[y_target_index] = 0

                if avg_loss_acc[y_target_index] >= self.attack_succ_threshold:
                    cost_up_counter[y_target_index] += 1
                    cost_down_counter[y_target_index] = 0
                else:
                    cost_up_counter[y_target_index] = 0
                    cost_down_counter[y_target_index] += 1

                if cost_up_counter[y_target_index] >= self.patience:
                    cost_up_counter[y_target_index] = 0
                    self.cost[y_target_index] *= self.cost_multiplier_up
                    self.cost_tensor[y_target_index] =  self.cost[y_target_index]
                    cost_up_flag[y_target_index] = True
                elif cost_down_counter[y_target_index] >= self.patience:
                    cost_down_counter[y_target_index] = 0
                    self.cost[y_target_index] /= self.cost_multiplier_down
                    self.cost_tensor[y_target_index] =  self.cost[y_target_index]
                    cost_down_flag[y_target_index] = True
           


            tmp_tensor = torch.Tensor(early_stop_tag)
            index = (tmp_tensor == False).nonzero()[:,0]
            time_tensor = torch.Tensor(times)[index]
            #print(time_tensor)
            non_early_stop_index = index
            non_opt_index = (time_tensor == 0).nonzero()[:,0]

            if early_stop_tag[y_target_index] == True and len(non_opt_index) != 0:
                for i in range(len(times)):
                    if times[i] == 0 and early_stop_tag[i] == False:
                        y_target_index = i
                        break

            elif len(non_opt_index) == 0 and early_stop_tag[y_target_index] == True:


                if len(non_early_stop_index)!= 0:
                    y_target_index = non_early_stop_index[torch.randint(0,len(non_early_stop_index),(1,)).item()]
                else:
                    break
            else: 
                if update[y_target_index] and times[y_target_index] >= self.warmup_rounds and all(ele >= self.warmup_rounds for ele in time_tensor):
                    self.early_stop = True
                    select_label = torch.max(torch.Tensor(reg_down_vel) + self.beta / torch.Tensor(avg_loss_reg),0)[1].item()
                    
                    random_value = torch.rand(1).item()


                    if random_value < self.epsilon_for_bandits:

                        non_early_stop_index = (torch.Tensor(early_stop_tag) != True).nonzero()[:,0]
                        
                        
                        if len(non_early_stop_index) > 1:
                            y_target_index = non_early_stop_index[torch.randint(0,len(non_early_stop_index),(1,)).item()]


                    else:
                        y_target_index = select_label

                elif times[y_target_index] < self.warmup_rounds  or update[y_target_index] == False:
                    continue

                else:
                    y_target_index = np.where(np.array(best_reg) == 1e+10)[0][0]


            #print('L1 of best_mask for each label:',best_reg)
            #print('L1 down speed:',reg_down_vel)
            #print('second loss item:',(1e+4 / torch.Tensor(avg_loss_reg)))
            #print('-----')
        return best_pattern, best_mask,best_reg,total_times
