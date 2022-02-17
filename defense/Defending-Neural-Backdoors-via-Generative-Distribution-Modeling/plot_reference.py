from matplotlib import offsetbox

def plot_annotationbox(ax, X, X0, h, w):
    shown_images = np.array([[1., 1.]])  # just something big
    for i in range(X.shape[0]):
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 3e-4:
            # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [X[i]]]
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(X0[i].reshape(3,h,w).transpose(1,2,0),
                                    zoom=5),
            X[i], frameon=False)
        ax.add_artist(imagebox)

def plot_embedding(sample, embedding, ax=None):
    X0, X = sample, embedding
#     x_min, x_max = np.min(X, 0), np.max(X, 0)
#     X = (X - x_min) / (x_max - x_min)

    if ax == None:
        plt.figure(figsize=(5,4))
        ax = plt.subplot(111)
        plot_annotationbox(ax, X, X0, h=3, w=3)
#         plt.xticks([]), plt.yticks([])
    else:
        plot_annotationbox(ax, X, X0, h=3, w=3)
        
import pickle

a=[]
b=[]
count=1
for i in range(10):
    a = a + [torch.tensor(np.load('random_trigger'+str(i)+'.npy')).view(-1).repeat(1, 1).float().numpy()]
    b = b + [(count,-0.05)]
    count = count + 1

trigger_target = [51, 90, 96, 79, 8, 50, 19, 7, 7, 91]

result_rounds = []
total = 10
for round in range(total):
    with open('results_rand_round_'+ str(round) +'.pickle', 'rb') as file:
        result_rounds = result_rounds + [pickle.load(file)]
        

with open('results_0.9_rand.pickle', 'rb') as file:
    results_90 = pickle.load(file)
with open('results_0.8_rand.pickle', 'rb') as file:
    results_80 = pickle.load(file)
with open('results_0.5_rand.pickle', 'rb') as file:
    results_50 = pickle.load(file)
dis_90 = []
dis_80 = []
dis_50 = []
baseline0=[]
baseline1=[]
baseline2=[]
baseline3=[]
baseline4=[]
baseline5=[]
baseline6=[]
baseline7=[]
baseline8=[]
baseline9=[]

for i in range(10):
    dis_90 = dis_90 + [results_90[i][0]['Test ASR'][-1]]
    dis_80 = dis_80 + [results_80[i][0]['Test ASR'][-1]]
    dis_50 = dis_50 + [results_50[i][0]['Test ASR'][-1]]
    baseline0 += [result_rounds[0][i][0]['Test ASR'][-1]]
    baseline1 += [result_rounds[1][i][0]['Test ASR'][-1]]
    baseline2 += [result_rounds[2][i][0]['Test ASR'][-1]]
    baseline3 += [result_rounds[3][i][0]['Test ASR'][-1]]
    baseline4 += [result_rounds[4][i][0]['Test ASR'][-1]]
    baseline5 += [result_rounds[5][i][0]['Test ASR'][-1]]
    baseline6 += [result_rounds[6][i][0]['Test ASR'][-1]]
    baseline7 += [result_rounds[7][i][0]['Test ASR'][-1]]
    baseline8 += [result_rounds[8][i][0]['Test ASR'][-1]]
    baseline9 += [result_rounds[9][i][0]['Test ASR'][-1]]
baseline = np.array([baseline0, baseline1, baseline2, baseline3, baseline4, baseline5, baseline6, baseline7, baseline8, baseline9])
ave = np.mean(baseline,axis=0)
std = np.std(baseline,axis=0)
baseline_ranking_index = np.argsort(ave).astype(int)
a = np.array(a)[baseline_ranking_index] 
plot_embedding(a,np.array(b))

plt.plot(range(1,11), np.array(dis_90)[baseline_ranking_index], label='distribution 0.9', marker='o', c='red', alpha=.5)
plt.plot(range(1,11), np.array(dis_80)[baseline_ranking_index], label='distribution 0.8', marker='o', c='green', alpha=.5)
plt.plot(range(1,11), np.array(dis_50)[baseline_ranking_index], label='distribution 0.5', marker='o', c='navy', alpha=.5)
plt.plot(range(1,11), ave[baseline_ranking_index], label = 'baseline', marker='o',c='gray', alpha=.5)

plt.plot(range(1,11), ave[baseline_ranking_index]+std[baseline_ranking_index], ls='--', c='gray')
plt.plot(range(1,11), ave[baseline_ranking_index]-std[baseline_ranking_index], ls='--', c='gray')
# for i in range(10):
#     plt.scatter(5,baseline4[i])
plt.fill_between(range(1,11), ave[baseline_ranking_index]-std[baseline_ranking_index], ave[baseline_ranking_index]+std[baseline_ranking_index], color = 'gray', alpha = .1)
# plt.plot(range(1,11), round1, label='one-point 2', marker = 'o', c='black', alpha=.5)
plt.legend(fontsize=14, frameon=False)
plt.xticks(range(1,11),fontsize=14)
plt.yticks([-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],fontsize=14)
plt.xlabel('10 random patterns on Cifar100' , fontsize=14)
plt.ylabel('ASR after defense', fontsize=14)

plot_embedding(a,np.array(b)+0.06-0.0099)
plt.plot(range(1,11), np.array(dis_90)[baseline_ranking_index], label='distribution 0.9', marker='o', c='red', alpha=.5)
plt.plot(range(1,11), np.array(dis_80)[baseline_ranking_index], label='distribution 0.8', marker='o', c='green', alpha=.5)
plt.plot(range(1,11), np.array(dis_50)[baseline_ranking_index], label='distribution 0.5', marker='o', c='navy', alpha=.5)
plt.legend(fontsize=14, frameon=False)
plt.xticks(range(1,11),fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('10 random patterns on Cifar100' , fontsize=14)
plt.ylabel('ASR after defense', fontsize=14)
