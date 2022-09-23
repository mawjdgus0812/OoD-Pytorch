import matplotlib.pyplot as plt
import seaborn as sns

def make_figure(name, clean_logits, ood_logits):
    plt.figure(figsize=(20,10))
    sns.histplot(np.max(clean_logits,axis=1), bins=100, label=f'Clean', color='tab:blue', legend=True)
    sns.histplot(np.max(ood_logits,axis=1), bins=100, label=f'OoD', color='tab:red', alpha=0.6, legend=True)


    plt.rc('font', size=20)        # 기본 폰트 크기
    plt.rc('axes', labelsize=30)   # x,y축 label 폰트 크기
    plt.rc('xtick', labelsize=30)  # x축 눈금 폰트 크기 
    plt.rc('ytick', labelsize=30)  # y축 눈금 폰트 크기
    plt.rc('legend', fontsize=30)  # 범례 폰트 크기
    # plt.rc('figure', titlesize=50) # figure title 폰트 크기

    plt.legend()
    plt.grid(True) 
    plt.xlabel(f'prob')
    plt.ylabel('Counts')
    plt.savefig(f'{name}')
    plt.close()
