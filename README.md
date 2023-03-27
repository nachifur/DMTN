# DMTN

<img width="512" height="512" src="https://github.com/nachifur/DMTN/blob/main/img/fig1.jpg"/>

# 1. Resources

## 1.1 Dataset
* SRD ([github](https://github.com/Liangqiong/DeShadowNet) | [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qu_DeshadowNet_A_Multi-Context_CVPR_2017_paper.pdf))
* ISTD ([github](https://github.com/DeepInsight-PCALab/ST-CGAN) | [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Stacked_Conditional_Generative_CVPR_2018_paper.pdf))
* ISTD+DA, SRD+DA ([github](https://github.com/vinthony/ghost-free-shadow-removal) | [paper](https://arxiv.org/abs/1911.08718))
* [SSRD](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EckYI_84wMdJpVgf5EqkmSABmnOD_-53YZ6v2KIQsiLeXA?e=67Gj47)

The SSRD dataset does not contain the ground truth of shadow-free images due to the presence of self shadow in images.

## 1.2 Results | Model Weight

**TEST RESULTS ON SRD:**
* Results on SRD: [DMTN_SRD](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EQt3ZoJAbZ5Cq_mhHxzrUVYBcsiaPLjnsN-SmhotYz-UOg?e=hS2RkJ) | Weight: [DMTN_SRD.pth](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EWc4B9PP-rtGp4LxPWGOkfoB6oi6Coh1tu-qG5qxBk-7Cg?e=sdlSiA)
* Results on SRD: [DMTN+Mask_SRD](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EU0JfEPuOUNNlSQDc0TfYgQBOAtMpXjK5yRoa3q2H_bcnQ?e=MgsEu5) | Weight: [DMTN+Mask_SRD.pth](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EU4NJ0CPbwpBrzyXH5FLlMEBzqwKhcXlxe8k4vQiXRrJUw?e=FPKlfF)

**TEST RESULTS ON ISTD:**
* Results on ISTD: [DMTN_ISTD](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EbyzaV72N2FElC5nOsp3-ZYBuUoVLiy29rmXBMXVXXY6Lg?e=wKA55D) | Weight: [DMTN_ISTD.pth](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EROUwnLgz9BGi3OtJa5SIs8BwdgYBZTXeMJ1NLcGfHAwCg?e=v1f51U)
* Results on ISTD: [DMTN+Mask_ISTD](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EcRgA4y1UAZIkRIabbm71iIBNhRH-JIugQDbInyWE3rpNQ?e=D8BiGD) | Weight: [DMTN+Mask_ISTD.pth](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EcnDQNKeoRdBtUYjQdirl34BR73n--qRnFIo6RnxPvk-KQ?e=9V0LIR)
* Results on ISTD+DA: [DMTN_ISTD_DA](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/ERGASEyFybBDm9rYZv4a3I4B6FwMmrhZMImk_-b7Lo-YeQ?e=MbzrMk) | Weight: [DMTN_ISTD_DA.pth](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EWwqrUr7Qh9KugvJ2S5KsdMBYz6aiR-ufiX3kn3zB626lg?e=7QJRra)

**TEST RESULTS ON ISTD+:**
* Results on ISTD+: [DMTN_ISTD+](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EZnB81g7L3VPuGo2zhVclVEBPhsO6MBJYPtbOnqxmEDHuw?e=MZLmUM) | Weight: [DMTN_ISTD+.pth](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/ETVno1MtDsdLknqDNKq60VwB9Bq-oq8kZ8B8aiwQBZXbQQ?e=B0S37N)
* Results on ISTD+: [DMTN+Mask_ISTD+](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EZEQr_hD7XdGgPiesl0L8aABSugt0z5U6V9q2Wv-fEr-VA?e=zq5A7s) | Weight: [DMTN+Mask_ISTD+.pth](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/ETo6UMeCGNhFjJ20o0RedaQBG7XIDcfbqucJ3A-hK6IQAQ?e=uN6sTs)

**TEST RESULTS ON SSRD:**

* Results of DMTN on SSRD: [DMTN_SSRD](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/ET7vtW6b-RNFiK7hJe8coXoBjMMUj2vZ4nEj1SitH8wuKA?e=ZDnfYV) | Weight: [DMTN_SRD_420_320.pth](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EQgZbEFJCLZGiAM8rnbE-ZUBHXw3zyTrhdet7JDSCrYiuA?e=6PcofV)
* Results of DHAN on SSRD:[DHAN_SSRD](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EROyGJwa2C5JkO1bLDVV_AsBbXRPKoZbBy5EsjAsz6xujg?e=nw53O6)

DHAN and DMTN are pretrained on SRD dataset (size:420x320)

## 1.3 Visual results


*Visual comparison results of **penumbra** removal on the SRD dataset - (Powered by [MulimgViewer](https://github.com/nachifur/MulimgViewer))*

<img src="https://github.com/nachifur/DMTN/blob/main/img/fig2.jpg"/>

*Visual comparison results of **self shadow** removal on the SSRD dataset - (Powered by [MulimgViewer](https://github.com/nachifur/MulimgViewer))*

<img src="https://github.com/nachifur/DMTN/blob/main/img/fig3.jpg"/>


## 1.4 Evaluation Code
Currently, MATLAB evaluation codes are used in most state-of-the-art works for shadow removal.

[Our evaluation code](https://github.com/nachifur/DMTN/blob/main/evaluation/evaluate_MAE_PSNR_SSIM.m) (i.e., 1+2)
1. MAE (i.e., RMSE in paper): https://github.com/tsingqguo/exposure-fusion-shadow-removal
2. PSNR+SSIM: https://github.com/zhuyr97/AAAI2022_Unfolding_Network_Shadow_Removal/tree/master/codes

Notably, there are slight differences between the different evaluation codes.
* [wang_cvpr2018](https://github.com/DeepInsight-PCALab/ST-CGAN), [le_iccv2019](https://github.com/cvlab-stonybrook/SID): no imresize;
* [fu_cvpr2021](https://github.com/tsingqguo/exposure-fusion-shadow-removal): first imresize, then double;
* [zhu_aaai2022](https://github.com/zhuyr97/AAAI2022_Unfolding_Network_Shadow_Removal): first double, then imresize;
* Our evaluation code: MAE->fu_cvpr2021, psnr+ssim->zhu_aaai2022

# 2. Environments
**ubuntu18.04+cuda10.2+pytorch1.7.1**
1. create environments
```
conda env create -f install.yaml
```
2. activate environments
```
conda activate DMTN
```

# 3. Data Processing
For example, generate the dataset list of ISTD:
1. Download:
   * ISTD and SRD
   * [USR shadowfree images](https://github.com/xw-hu/Mask-ShadowGAN)
   * [Syn. Shadow](https://github.com/vinthony/ghost-free-shadow-removal)
   * [SRD shadow mask](https://github.com/vinthony/ghost-free-shadow-removal)
   * train_B_ISTD:
   ```
   cp -r ISTD_Dataset_arg/train_B ISTD_Dataset_arg/train_B_ISTD
   cp -r ISTD_Dataset_arg/train_B SRD_Dataset_arg/train_B_ISTD
   ```
   * [VGG19](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)
   ```
   cp vgg19-dcbb9e9d.pth ISTD_Dataset_arg/
   cp vgg19-dcbb9e9d.pth SRD_Dataset_arg/
   ```
2. The data folders should be:
    ```
    ISTD_Dataset_arg
        * train
            - train_A # ISTD shadow image
            - train_B # ISTD shadow mask
            - train_C # ISTD shadowfree image
            - shadow_free # USR shadowfree images
            - synC # Syn. shadow
            - train_B_ISTD # ISTD shadow mask
        * test
            - test_A # ISTD shadow image
            - test_B # ISTD shadow mask
            - test_C # ISTD shadowfree image
        * vgg19-dcbb9e9d.pth

    SRD_Dataset_arg
        * train #  renaming the original `Train` folder in `SRD`.
            - train_A # SRD shadow image, renaming the original `shadow` folder in `SRD`.
            - train_B # SRD shadow mask
            - train_C # SRD shadowfree image, renaming the original `shadow_free` folder in `SRD`.
            - shadow_free # USR shadowfree images
            - synC # Syn. shadow
            - train_B_ISTD # ISTD shadow mask
        * test #  renaming the original `test_data` folder in `SRD`.
            - train_A # SRD shadow image, renaming the original `shadow` folder in `SRD`.
            - train_B # SRD shadow mask
            - train_C # SRD shadowfree image, renaming the original `shadow_free` folder in `SRD`.
        * vgg19-dcbb9e9d.pth 
    ```
3. Edit `generate_flist_istd.py`: (Replace path)

```
ISTD_path = "/Your_data_storage_path/ISTD_Dataset_arg"
```
4. Generate Datasets List. (Already contains ISTD+DA.)
```
conda activate DMTN
cd script/
python generate_flist_istd.py
```
5. Edit `config_ISTD.yml`: (Replace path)
```
DATA_ROOT: /Your_data_storage_path/ISTD_Dataset_arg
```

# 4. Training+Test+Evaluation
## 4.1 Training+Test+Evaluation
For example, training+test+evaluation on ISTD dataset.
```
cp config/config_ISTD.yml config.yml 
cp config/run_ISTD.py run.py
conda activate DMTN
python run.py
```
## 4.2 Only Test and Evaluation
For example, test+evaluation on ISTD dataset.
1. Download weight file(`DMTN_ISTD.pth`) to `pre_train_model/ISTD`
2. Copy file
```
cp config/config_ISTD.yml config.yml 
cp config/run_ISTD.py run.py
mkdir -p checkpoints/ISTD/
cp config.yml checkpoints/ISTD/config.yml
cp pre_train_model/ISTD/DMTN_ISTD.pth  checkpoints/ISTD/ShadowRemoval.pth
```

3. Edit `run.py`. Comment the training code.

```
    # # pre_train (no data augmentation)
    # MODE = 0
    # print('\nmode-'+str(MODE)+': start pre_training(data augmentation)...\n')
    # for i in range(1):
    #     skip_train = init_config(checkpoints_path, MODE=MODE,
    #                             EVAL_INTERVAL_EPOCH=1, EPOCH=[90,i])
    #     if not skip_train:
    #         main(MODE, config_path)
    # src_path = Path('./pre_train_model') / \
    #     config["SUBJECT_WORD"]/(config["MODEL_NAME"]+'_pre_da.pth')
    # copypth(dest_path, src_path)

    # # train
    # MODE = 2
    # print('\nmode-'+str(MODE)+': start training...\n')
    # for i in range(1):
    #     skip_train = init_config(checkpoints_path, MODE=MODE,
    #                             EVAL_INTERVAL_EPOCH=0.1, EPOCH=[60,i])
    #     if not skip_train:
    #         main(MODE, config_path)
    # src_path = Path('./pre_train_model') / \
    #     config["SUBJECT_WORD"]/(config["MODEL_NAME"]+'_final.pth')
    # copypth(dest_path, src_path)
```
4. Run

```
conda activate DMTN
python run.py
```
## 4.3 Show Results
After evaluation, execute the following code to display the final RMSE.
```
python show_eval_result.py
```
Output:
```
running rmse-shadow: xxx, rmse-non-shadow: xxx, rmse-all: xxx # ISRD
```
This is the evaluation result of python+pytorch, which is only used during training. To get the evaluation results in the paper, you need to run the [matlab code](#1.4).

## 4.4 Test on SSRD
1. Edit `src/network/network_DMTN.py`. Modify the line (https://github.com/nachifur/DMTN/blob/main/src/network/network_DMTN.py#L339).
```
SSRD = True
```
2. Test like the section `4.2 Only Test and Evaluation`.

# 5. Acknowledgements
Part of the code is based upon:
* https://github.com/nachifur/LLPC
* https://github.com/vinthony/ghost-free-shadow-removal
* https://github.com/knazeri/edge-connect

# 6. Citation
```
@ARTICLE{liu2023decoupled,
  author={Liu, Jiawei and Wang, Qiang and Fan, Huijie and Li, Wentao and Qu, Liangqiong and Tang, Yandong},
  journal={IEEE Transactions on Multimedia}, 
  title={A Decoupled Multi-Task Network for Shadow Removal}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TMM.2023.3252271}}
```
# 7. Contact
Please contact Jiawei Liu if there is any question (liujiawei18@mails.ucas.ac.cn).