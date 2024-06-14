
# subset train
train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
clean_dataset = self.result['clean_train'].wrapped_dataset
data_all_length = len(clean_dataset)
ran_idx = choose_index(self.args, data_all_length) 
log_index = self.args.log + 'index.txt'
np.savetxt(log_index, ran_idx, fmt='%d')
clean_dataset.subset(log_index)
data_set_without_tran = clean_dataset
data_set_o = self.result['clean_train']
data_set_o.wrapped_dataset = data_set_without_tran
data_set_o.wrap_img_transform = train_tran

# no subset train
train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
data_set_o = self.result['clean_train']
data_set_o.wrap_img_transform = train_tran

# subset test

# no subset test
test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
data_set_o = self.result['clean_test']
data_set_o.wrap_img_transform = test_tran