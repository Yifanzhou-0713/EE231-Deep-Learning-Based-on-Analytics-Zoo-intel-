## Code Structure
LPRNet
				data
								load_data.py                                                         导入图片
								preprocess.py                                                      图片预处理
								NotoSansCJK-Regular.ttc
				model                                                                                    网络代码
								LPRNET.py
								STN.py
				weights                                                                                 保存LPRNet和STN的训练模型
				Evaluation.py
				LPRNet_Train.py                                                                  LPRNet模型训练
				STN_Train.py                                                                        STN模型训练
				LPRNet_Test.py                                                                    LPRNet车牌信息识别demo

MTCNN
				data_preprocessing
								assemble_Onet_imglist.py                                  生成Onet图片路径
								assemble_Pnet_imglist.py                                   生成Pnet图片路径
								gen_Pnet_train_data.py                                       获得预处理后图片
								gen_Onet_train_data.py                                       获得预处理+Pnet后图片
				model                                                                                      网络代码
								MTCNN_nets.py
				train
								Data_Loading.py                                                    导入图片
								Train_Onet.py                                                         Onet模型训练
								Train_Pnet.py                                                         Pnet模型训练
				utils
								util.py                                                                       一些计算函数
				weights                                                                                    保存Onet和Pnet的训练模型
				MTCNN.py                                                                               MTCNN车框识别demo
				test                                                                                           demo用测试图片
				main.py                                                                                    车牌识别demo
				
				

## Training on MTCNN
* run 'MTCNN/data_set/preprocess.py' to split training data and validation data and put in "ccpd_train" and "ccpd_val" folders respectively.
* run 'MTCNN/data_preprocessing/gen_Pnet_train_data.py', 'MTCNN/data_preprocessing/gen_Onet_train_data.py','MTCNN/data_preprocessing/assemble_Pnet_imglist.py', 'MTCNN/data_preprocessing/assemble_Onet_imglist.py' for training data preparation.
* run 'MTCNN/train/Train_Pnet.py' and 'MTCNN/train/Train_Onet.py
## Training on LPRNet
* run 'LPRNet/data/preprocess.py' to prepare the dataset
* run 'LPRNet/LPRNet_Train.py' for training 

## Test
* run 'MTCNN/MTCNN.py' for license plate detection
* run 'LPRNet/LPRNet_Test.py' for license plate recognition
* run 'main.py' for both
## Reference
* [MTCNN](https://arxiv.org/abs/1604.02878v1)
* [LPRNet](https://arxiv.org/abs/1806.10447)
* [Spatial Transformer Layer](https://arxiv.org/abs/1506.02025)
* [LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch)