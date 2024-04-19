Additional Information for CHIL Submission: Multiple Instance Learning with Absolute Position Information

File Definitions

For All Tasks
- hyperparameters.txt: final hyperparameters and the hyperparameters searched over for all MIL methods on all datasets

For Real Data Tasks
- pretrain_chexpert.txt: names of the images we used to pretrain models that evaluate our real data task of classifying cardiomegaly in chest x-rays
- train_mimic.txt, val_mimic.txt, test_mimic.txt: names of images in our training, validation, and test splits of the models that evaluate our real data task of classifying cardiomegaly in chest x-rays
- train_chexpert_edema.txt: names of the images we used to pretrain models that evaluate our real data task of classifying pulmonary edema in chest x-rays
- train_mimic_edema.txt, val_mimic_edema.txt, test_mimic_edema.txt: names of images in our training, validation, and test splits of the models that evaluate our real data task of classifying pulmonary edema in chest x-rays

For Synthetic Data Tasks
- classification_times.txt: classification times of all methods on the synthetic data task

Folder Definitions
- MNIST_Ordered_List_Task: jupyter notebooks to create the ordered lists in the MNIST_Ordered_List_Task
- SyntheticData_ModelsTraining, RealData_ModelsTraining: code to train the models/ obtain results for synthetic and real data

