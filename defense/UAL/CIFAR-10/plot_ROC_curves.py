import matplotlib.pyplot as plt
import pickle

# load ROC information
fpr_list=list()
tpr_list=list()
auc_list=[0.5]

for N in [1, 5, 10]:
	[fpr, tpr, thresholds, auc] = pickle.load(open("./results/ROC_ULP_N{}.pkl".format(N), "rb"))
	fpr_list.append(fpr)
	tpr_list.append(tpr)
	auc_list.append(auc)

for N in [1, 5, 10]:
	[fpr, tpr, thresholds, auc] = pickle.load(open("./results/ROC_Noise_N{}.pkl".format(N), "rb"))
	fpr_list.append(fpr)
	tpr_list.append(tpr)
	auc_list.append(auc)


legend=['Chance','ULP - M=1','ULP - M=5','ULP - M=10','Noise - M=1','Noise - M=5','Noise - M=10']
legends=[leg+', AUC=%0.02f'%(auc) for (auc,leg) in zip(auc_list,legend)]
plt.figure(figsize=(7,5.5))
plt.plot([0, 1], [0, 1], linestyle=':',linewidth=3)
for i in range(6):
	# plot the roc curve for the model
	if i<3:
		plt.plot(fpr_list[i], tpr_list[i],linewidth=3)
	else:
		plt.plot(fpr_list[i], tpr_list[i],'--',linewidth=3)
plt.legend(legends,fontsize=14)
plt.xticks(fontsize=14), plt.yticks(fontsize=14)
plt.xlabel('1-Specificity',fontsize=16)
plt.ylabel('Sensitivity',fontsize=16)
plt.title('CIFAR-10 - VGG Mod',fontsize=16)
# show the plot
plt.show()