#smote..구현

import numpy as np

def smote_oversampling(X, y, k=5, oversample_ratio=0.5): #1:1 ratio
    #우울증 환자(1), 우울증 아닌 사람(0)
    minority_class_indices = np.where(y==1)[0]
    majority_class_indices = np.where(y==0)[0]
    
    num_minority_samples = len(minority_class_indices)
    num_majority_samples = len(majority_class_indices)
    
    
    #number of synthetics samples
    '''
    1:1 비율을 기준으로 소수 클래스 목표 샘플수 계산 후, 현재 샘플 수와 비교해서 필요한 만큼 새롭게 데이터 수를 결정
    synthetics samples는 무작위로 선택한 minor class의 샘플과 그 주변 이웃을 이용하여 생성함
    '''
    target_minority_samples = int(num_majority_samples * oversample_ratio)
    num_synthetic_samples = max(0, target_minority_samples - num_minority_samples)
    
    synthetic_samples = []
    
    for i in range(num_synthetic_samples):
        random_minority_index = np.random.choice(minority_class_indices)
        target_sample = X[random_minority_index]
        
        knn_indices = np.argsort(np.linalg.norm((X - target_sample), axis=1))[1:k+1]
        selected_neighbor_index = np.random.choice(knn_indices)
        
        alpha = np.random.rand()
        
        synthetic_sample = target_sample + alpha * (X[selected_neighbor_index] - target_sample)
        synthetic_samples.append(synthetic_sample)
        
    synthetic_samples = np.array(synthetic_samples)
    
    #binary classification - 0,1
    synthetic_labels = np.ones((num_synthetic_samples,))
    
    X_resampled = np.vstack([X, synthetic_samples])
    y_resampled = np.concatenate([y, synthetic_labels])
    
    return X_resampled, y_resampled

#X_resampled, y_resampled = smote_oversampling(X, y, k=5, oversample_ratio=0.5)

