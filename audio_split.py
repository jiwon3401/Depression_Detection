import os
import shutil
import pandas as pd

# folder path setting
origin_data = '/home/user/sevenpointone/DAIC-WOZ'
audio_folder = os.path.join(origin_data, "audio_files")
csv_folder = os.path.join(origin_data, 'labels')

# load train,val,test file
train_csv = pd.read_csv(os.path.join(csv_folder, 'train_split.csv'))
val_csv = pd.read_csv(os.path.join(csv_folder, 'dev_split.csv'))
test_csv = pd.read_csv(os.path.join(csv_folder, 'test_split.csv'))

# 각 split에 대해 폴더 생성하고 오디오 파일 복사
def audio_split():
    for split_name, split_csv in [('train', train_csv), ('val', val_csv), ('test', test_csv)]:
        destination_folder = os.path.join(audio_folder, f'{split_name}_split')
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder, exist_ok=True)
            
        for _, row in split_csv.iterrows():
            audio_id = row['Participant_ID']
            audio_file_path = os.path.join(audio_folder, f'{audio_id}_AUDIO.wav')
            shutil.copy2(audio_file_path, destination_folder)


if __name__=="__main__":
    audio_split()
    print("작업 완료!")