# :busts_in_silhouette:BothGAN

<p align="center"><img src="https://github.com/user-attachments/assets/9ec0f771-36c3-45a0-9d61-ca58d02e2705"></p>


## 프로젝트 소개
- 중성적인 사람 얼굴을 생성하기 위한 이중 판별자를 활용한 생성적 적대 신경망(GAN)의 변형 모델을 제안한다.
- 생성자를 통해 가짜 이미지가 생성되고, 이를 두 가지 판별자에게 동시에 비교되면서 두 도메인의 특성을 동시에 학습한다.
- 최종적으로는 남성과 여성의 특징을 동시에 가진 중성적인 얼굴이 생성된다.
- 모델의 성능비교를 위해 남성과 안경 낀 사람으로 추가 테스트하여, 결과적으로 안경 낀 남성이 생성되는지 확인한다.

## 개발 환경
- **언어**: python
- **tool**:
  - jupyter
  - tensorflow == 2.10.0
  - cudatoolkit == 11.7.0
  - cuDnn == 8.4
- **dataset**: [celebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

## 보고서
![image](https://github.com/user-attachments/assets/fc89dce2-468d-4942-bf09-99d4b94ef63d)

![image](https://github.com/user-attachments/assets/df87f385-074a-4004-8ef7-ace286e9bf1d)

![image](https://github.com/user-attachments/assets/2455916c-bc5e-4c7e-889a-271d600ee92a)
