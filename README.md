# 2021-1 SWCON Capstone Design Project

해당 레포지토리는 경희대학교 2021학년도 소프트웨어융합캡스톤디자인 수업의 프로젝트 결과물을 포함하고 있습니다. 발표 동영상을 시청하고 싶으시다면 [이 링크](https://drive.google.com/file/d/1ckvnqR6i_PHnUG0j_sVdyzaINsEw1VFE/view?usp=sharing)에 방문해주시길 바랍니다.


# 1. 과제 개요
## 1.1 과제 선정 배경 및 필요성[1]
2018년 미국 커뮤니티 사이트 레딧(Reddit)에서 ‘딥 페이크(DeepFakes)’라는 이름의 유저가 처음으로 제시한 신경망 기반 얼굴 영상 조작 프로그램이 현재 딥 페이크 기법으로 불리는 영상 조작 기술의 시초이다. 딥 페이크 기법은 의료, 교육 등 다양한 산업에서 긍정적으로 사용될 여지가 있음에도 불구하고, 불법 음란물 또는 가짜 뉴스 생성에 악용되고 있다. 현재 인터넷에 유포된 딥 페이크 영상의 약 96%가 불법 음란물 동영상인 것으로 밝혀져, 가짜 뉴스를 통한 선동 뿐만 아니라 딥 페이크 포르노 문제가 사회적인 이슈로 떠오르고 있다. 이는 포르노 영상에 특정인의 얼굴을 합성한 영상으로, 유명인을 넘어 일반인에게까지 그 대상이 확대되고 있다. 따라서 사람 얼굴 영역을 조작한 영상을 자동화된 기법으로 판별하고, 분류해낼 수 있는 시스템의 성능 개선이 필요하다 생각하여 주제로 선정하게 되었다.

## 1.2 과제 주요내용
### 1.2.1 사용하고자 하는 데이터 셋
분류 모델 학습에 Celeb-DF와 FaceForensics++(FF++)의 NeuralTextures(c40, LQ) 데이터를 사용하고자 한다. 이는 선행 연구를 참조하였을 때, 정확도가 낮은 편에 속하는 도전적인 데이터 셋이므로 이에 대한 성능 향상이 필요하다 생각하게 되었다.  

<p align="center">
<img src="./3. Figures/Table1.png"height="180px" width="494px" />
<br>
<b>Table 1.</b> 모델별 데이터 셋에 대한 AUC(%) 점수표[2].</br>
</p>
<br>
<p align="center">
<img src="./3. Figures/Table2.png"height="199px" width="386px"/>
<br>
<b>Table 2.</b> FF++ 전체 데이터 셋에 대한 모델별 정확도(accuracy) 도표[3]. Raw는 압축을 가하지 않은 원본 영상, HQ(c 20)는 압축의 정도가 약한 데이터, LQ(c 40)은 압축의 정도가 강한 데이터를 의미한다. 압축을 강하게 해 영상 데이터의 품질이 낮아진 LQ 데이터의 경우, 전체 조작 카테고리의 평균적인 정확도가 81%로, HQ와 Raw 데이터에 비해 조작 탐지가 어려움을 알 수 있다[3].</br>
</p>
</p>
<br>
<p align="center">
<img src="./3. Figures/Table3.png"height="200px" width="400px"/>
<br>
<b>Table 3.</b> 선행 연구 모델들을 이용한 FF++의 조작 카테고리별 조작 탐지 정확도 점수 도표[3]. 해당 도표의 모든 정확도는 LQ 데이터 셋에 대한 결과이다. 특히, NT를 사용하여 영상 속 사람 얼굴의 표정을 조작한 데이터에 대한 정확도(빨간 박스)는 대략 80% 정도로, 다른 조작 기법에 비해 도전적인 데이터 셋임을 알 수 있다.</br>
</p>
<br>
Table 1을 통해 Celeb-DF가 이전 세대 데이터 셋에 비해 조작 탐지가 어려운, 도전적인 데이터 셋임을 알 수 있다. 이전의 데이터 셋은 해상도 불일치와 같은 시각적 아티팩트가 명확해 매우 높은 성능을 달성했다. 하지만 Celeb-DF의 경우, 가장 높은 성능의 AUC 점수가 65.5에 불과하다.
<br></br>
 Table 2를 통해 FF++의 LQ 데이터가 탐지 성능이 낮음을 알 수 있고, 특히 Table 3을 통해 NT의 탐지가 어렵다는 것을 알 수 있다. 따라서 이번 프로젝트에서는 기존에 성능이 이미 좋게 측정되어 왔던 DeepFakes, Face2Face, FaceSwap의 raw 또는 HQ 데이터 보다는, Celeb-DF와 NT의 LQ 데이터에 집중하여 모델 성능을 향상시키고자 한다.
 실제 SNS 상에 유포되는 조작 영상들은 압축된 상태로 배포되기 때문에 조작 여부 예측에 어려움이 있을 수 있다. 따라서 본 시스템의 훈련과 평가에서 사용될 데이터 셋 역시 고압축된 저해상도 영상(NT; LQ)과 최신 조작 기법을 사용해 육안으로도 상당히 자연스러워 보이는 고품질 영상(Celeb-DF)을 선정하였다.

### 1.2.2 시스템 개요
<p align="center">
<img src="./3. Figures/Figure1.png"height="120px" width="400px"/>
<br>
<b>Figure 1.</b> 네트워크에 데이터를 넣어주기 전 영상(mp4)에서 프레임(jpg)을 뽑아내는 과정. 영상을 입력하면 일정 개수(frame per video, fpv=25)의 연속된 프레임을 뽑아내고, MTCNN을 사용하여 얼굴 영역을 추출 및 정렬한다.</br>
</p>

<br></br>

<p align="center">
<img src="./3. Figures/Figure2.png"height="110px" width="400px"/>
<br>
<b>Figure 2.</b> Figure 1에서 처리한 프레임 데이터를 XceptionNet에 넣어주고, 이를 통해 추출한 features를 다시 Bidirectional-RNN(LSTM)에 넣어 처리할 계획이다. 일정 개수(fpv=25)의 연속된 프레임을 하나의 시퀀스로 처리한다.</br>
</p>

<br></br>

<p align="center">
<img src="./3. Figures/Table4.png"height="110px" width="400px"/>
<br>
<b>Table 4.</b> 얼굴 영상의 정렬과, 양방향 LSTM의 사용이 성능 향상에 기여를 하는지 실험한 연구의 실험 결과 도표[4]. </br>
</p>

영상에서 얼굴 영역을 추출 및 정렬하고, CNN 아키텍처에 양방향 RNN 아키텍처를 함께 사용하는 구조가 성능 향상에 도움이 되는 것을 증명한 연구[4]는 이미 존재한다. 해당 연구는 랜드마크 기반의 얼굴 정렬로 데이터를 전처리하고, DenseNet에 양방향 RNN을 함께 사용하여 FF++의 DF, F2F, FS에 대한 정확도를 최대 4.55% 향상시켰다[4]. 본 프로젝트는 이 선행 연구[4]의 실험 결과를 참고하여 Figure 1과 같이 데이터를 전처리하고, Figure 2와 같은 프로세스로 학습 시키는 방식을 채택했다. CNN 아키텍처를 이용하여 frame-level의 시각적 아티팩트를 이용하고, 양방향 RNN을 이용하여 sequence-level의 시간적 불일치성을 이용하여 영상의 조작 여부를 탐지한다.

<p align="center">
<img src="./3. Figures/Figure3.png"height="250px" width="370px"/>
<br>
<b>Figure 3.</b> FF++의 raw, HQ, LQ 데이터에 대한 모델별 정확도 결과 도표[3]. </br>
</p>

FF++를 구축 및 제안한 연구[3]에서 영상을 압축한 정도에 따라 조작 카테고리별로 데이터 셋을 구축했다. 대규모의 다양한 조작 기법별 얼굴 조작 영상 데이터 셋 공개 이외에도 얼굴 조작 영상 탐지에 대한 벤치마크를 제시하였는데, Figure 3을 통해 알 수 있듯이 XceptionNet이 가장 높은 성능을 보인다. 이는 후속 연구들에서 Xception 기반의 아키텍처를 채택하는 데에 크게 기여했다. Celeb-DF와 NeuralTextures(c40)의 state-of-the-art 성능을 달성한 최신 연구[5]에서 역시 XceptionNet을 사용하여 Celeb-DF에 대한 성능의 정확도 96%로 크게 향상시킨 바가 있다. 따라서 프로젝트 초기에 채택한 모델은 Keras에서 제공하는 Xception 모델이었다. 이후 프로젝트를 진행하던 도중 Xception 전이학습의 성능이 향상되지 않아 Figure 5를 참고하여 EfficientNet의 B7으로 변경하게 되었다. 이에 대해서는 2. 과제 수행방법 부분에서 자세히 설명한다. 

## 1.3 최종결과물의 목표
-  Celeb-DF와 FF++의 NT(c40) 데이터 셋에 대한 탐지 성능 정확도 90% 이상.
- 최신 연구의 성능과 비교했을 때, 성능을 향상시킬 것.
- 영상 하나를 입력했을 때, 조작되었을/조작되지 않았을 확률을 출력하는 일련의 시스템 구현.

# 2. 과제 수행 방법
## 2.1 과제 수행을 위한 도구적 방법
Python 3.5를 사용하여 구현했고, 사용한 오픈소스는 다음과 같다:
1. facenet-pytorch의 MTCNN
2. Keras의 pretrained model(Xception, EfficientNet-B7)
3. Tensorflow 1.x
4. OpenCV
5. Celeb-DF & FaceForensics++

## 2.2 과제 수행 과정
<p align="center">
<img src="./3. Figures/Figure4.png"height="200px" width="280px"/>
<br>
<b>Figure 4.</b> FF++의 benchmark 사이트에서 NeuralTextures의 정확도를 기준으로 정렬한 모습[6].</br>
</p>

Xception, EfficientNet-B7의 경우, keras application 모듈에서 제공하는 사전 훈련된 모델을 사용하였다. ImageNet에 대한 웨이트 값을 사용했고, 각각의 레이어에 대한 웨이트 값을 고정 해제하여 본 프로젝트에서 사용된 Celeb-DF와 NT에 대해 새롭게 학습하도록 했다. 프로젝트 초기에 CNN 모델로 Xception을 채택해 사용하였는데, 전이학습 코드 상의 문제점으로 인해 성능 향상이 되지 않는 문제가 발생했다. 새로운 코드를 사용하는 동시에 Figure 5를 참조하여 모델을 변경하였다. 따라서 새로운 전이학습 코드를 이용한 Xception과 EfficientNet에 대한 성능 비교를 추가로 진행할 계획이다.


<p align="center">
<img src="./3. Figures/Figure5.png"height="212px" width="530px"/>
<br>
<b>Figure 5.</b> 과제 수행의 전체 프로세스.</br>
</p>

전체적인 프로젝트의 프로세스는 Figure 5와 같다. Celeb-DF와 FaceForensics++의 NT 영상(mp4)에서 모든 프레임을 추출한다. 이렇게 추출한 프레임에서 MTCNN을 사용하여 얼굴 영역을 추출하고, 하나의 영상당 25개의 얼굴 영상을 무작위로 추출하여 모델을 학습시킬 데이터를 생성한다. 이 데이터를 사용하여 Keras의 ImageNet을 학습한 EfficientNet을 전이학습 시키고, 학습 결과(weight)를 저장한다. 다시 Celeb-DF와 NT 영상에서 프레임을 추출하고, 얼굴 영역을 추출한다. 이때는 LSTM을 학습시키기 위한 데이터 셋이므로, 하나의 영상에서 시간의 흐름에 맞게 25개의 프레임을 추출한다. 프레임에 대한 앞서 훈련시킨 EfficientNet 모델의 prediction 결과를 numpy 배열로 변환해 양방향 LSTM을 훈련시킨다.
<br></br>
EfficientNet은 앞서 언급했듯 ImageNet에 대한 weight 값을 초기 weight로 하고, 이후 학습을 통해 본 프로젝트의 데이터 셋에 대한 weight 값을 얻는다. 이 과정을 2번 반복하는데, 처음 학습된 weight 값을 불러들여 다시 학습을 진행한다. 이때, test set은 학습 이전에 분리시켜 학습 도중 함께 학습되는 경우가 없도록 하여 overfitting을 방지한다. 양방향 LSTM의 데이터 셋은 두 번째 학습시킨 EfficientNet(성능이 더 향상됨)을 사용하여 feature를 추출한다.

# 3. 수행 결과
## 3.1 과제 수행 결과
### 3.1.1 Celeb-DF 실험 결과
선행 연구[5]의 성능과 비교하기 위해 얼굴 영상 하나의 크기를 160x160으로 세팅하였고, 한 영상당 25개의 프레임을 사용했다. Celeb-DF 자체에 포함된 518개의 영상을 test set으로 사용했고, 이를 제외한 나머지에서 20%를 validation set으로 사용했다. 총 20 에포크를 학습시켰고, early stopping을 위한 patience는 5로 설정했다.

<p align="center">
<img src="./3. Figures/Table5.png"height="150px" width="1680px"/>
<br>
<b>Table 5.</b> Celeb-DF의 실험 결과 도표.</br>
</p>

선행 연구[5]에서의 validation set에 대한 정확도는 98%로, 본 프로젝트에서 최대 **2%**를 향상시켰다. Test set에 대한 정확도와 비교해도 최대 **1.8%**를 향상시켰다. Table 5를 통해 **양방향 LSTM**에서 validation set과 test set에 대한 정확도가 가장 높음을 알 수 있다.

### 3.1.2 NeuralTextures 실험 결과
선행 연구[3, 5]의 성능과 비교하기 위해 얼굴 영상 하나의 크기를 299x299으로 세팅하였고, 한 영상당 25개의 프레임을 사용했다. Celeb-DF 자체에 포함된 240개의 영상을 test set으로 사용했고, 이를 제외한 나머지에서 20%를 validation set으로 사용했다. 총 20 에포크를 학습시켰고, early stopping을 위한 patience는 5로 설정했다.

<p align="center">
<img src="./3. Figures/Table6.png"height="150px" width="1680px"/>
<br>
<b>Table 6.</b> NT의 실험 결과 도표.</br>
</p>

선행 연구[5]에서의 validation set에 대한 정확도는 90.71%로, 본 프로젝트에서 최대 **8%**를 향상시켰다. Test set에 대한 정확도와 비교해도 최대 **2%**를 향상시켰다. Table 6을 통해 validation set과 train set에 대한 정확도는 EfficientNet에서 더 높지만, test set에 대한 정확도는 **양방향 LSTM**에서 가장 높음을 알 수 있다.

## 3.2 최종 결과물 주요 특징 및 설명
Table 5와 6을 통해 알 수 있는 것은 Celeb-DF와 NeuralTextures 모두 양방향 LSTM에서 test 정확도가 가장 높다는 것이다. NeuralTextures의 경우, train set과 validation set에 대한 정확도는 EfficientNet만 사용했을 때가 더 높지만, 학습에 사용되지 않은 test set에 대한 성능은 양방향 LSTM에서 더 높은 것을 보아 일반화 성능은 양방향 LSTM이 더 뛰어남을 알 수 있다.
<br></br>
 위와 같은 결과를 통해 CNN 아키텍처만을 사용하지 않고, 양방향 LSTM을 사용하여 시간적인 연속성/불일치성을 확인하는 방식이 성능 향상에 도움이 된다는 것을 알 수 있다.

 # 4. 기대효과 및 활용 방안
 ## 4.1 기대 효과
Celeb-DF와 같이 최신 조작 기법을 사용한 정교한 조작 영상과, NeuralTextures의 LQ 데이터와 같이 표정만 조작한 저품질의 영상에 대한 성능을 크게 향상시켰다. 인터넷 상에서 실제 유포되는 조작 영상이 주로 SNS를 통해 퍼지는 점(데이터 압축이 많이 진행되어 품질이 떨어짐)과, 얼굴 표정을 조작하여 가짜 뉴스를 생성하는 경우가 많다는 점을 고려했을 때, 실제 인터넷 상의 조작 영상에 대한 좋은 탐지 성능을 기대할 수 있다.

 ## 4.2 활용 방안
 앞서 말했던 바와 같이, Xception 모델과의 성능 비교 실험을 추가로 진행하고, 실제 인터넷 상에 유포되는 조작 영상에 대한 탐지 결과를 확인하는 실험을 추가로 진행할 계획이다. 모델 사이즈를 조절한다면, 본 프로젝트의 프로세스를 활용하여 인터넷 상의 영상에 대한 자동화된 탐지 프로세스를 개발할 수 있을 것이다.

 # 5. 결론 및 제언
본 프로젝트에서 Celeb-DF에 대한 정확도는 최대 2%, NeuralTextures에 대한 정확도는 최대 8% 향상시켰다. 이를 통해 얼굴 영역 추출 및 정렬과 CNN 아키텍처 + 양방향 RNN 아키텍처가 실제로 성능 향상에 효과적임을 보일 수 있었다. 향후 epoch과 patience를 늘려서 여러 차례 CNN을 학습시키고, 가장 성능이 좋은 모델을 사용하여 feature를 추출하고 LSTM을 학습시키는 실험을 추가로 진행할 계획이다. 최종 결과물에는 EfficientNet-B7만을 사용하였지만, 기존의 조작 탐지 분야에서 높은 성능으로 주목 받았던 Xception을 추가로 사용하여 실험을 진행하고, 이 둘을 비교할 계획이다. 추가적인 향후 목표로는, faceforensics++ benchmark 사이트[6]에 향상된 성능의 모델을 제출하는 것을 생각하고 있다.

 # 6. References
 [1] Kim, J., An, J., Yang, B., Jung, J., & Woo, S. S. (2020). 데이터 기반 딥페이크 탐지기법에 관한 최신 기술 동향 조사. Review of KIISC, 30(5), 79-92.

[2]Li, Y., Yang, X., Sun, P., Qi, H., & Lyu, S. (2019). “Celeb-ㄴDF (v2): a new dataset for DeepFake Forensics.” arXiv preprint arXiv:1909.12962.

[3] Rossler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). “Faceforensics++: Learning to detect manipulated facial images.” In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 1-11).

[4] Sabir, E., Cheng, J., Jaiswal, A., AbdAlmageed, W., Masi, I., & Natarajan, P. (2019). “Recurrent convolutional strategies for face manipulation detection in videos.” Interfaces (GUI), 3(1). 

[5] Kumar, A., Bhavsar, A., & Verma, R. (2020, April). “Detecting deepfakes with metric learning.” In 2020 8th International Workshop on Biometrics and Forensics (IWBF) (pp. 1-6). IEEE.

[6] http://kaldir.vc.in.tum.de/faceforensics_benchmark/ 
