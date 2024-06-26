![seoultech](https://github.com/ChangsunLeee/HW2/assets/167077784/a93d1135-86f0-4205-a140-2092311546d2)<br/><br/>
24510056_기계설계로봇공학과_석사과정_이창선 <br/>
파일구성 : dataset.py, model.py, main.py, main.py, LeNet5.png, CustomMLP.png <br/>
     (model.py와 model2.py의 차이점: model.py에서 LeNet-5의 기본 형식을 구현하려 했으며, model2.py에서 정규화기법을 사용하여 모델을 개선) <br/>
     (model2.py의 경우 main.py의 7 line model -> model2로 변경한 뒤 선언하고 사용해야함.) <br/>
실행프로그램 : Visual Studion Code <br/>
<br/>
<br/>
1.LeNet5와 CustomMLP의 파라미터 갯수 <br/>
  LeNet5: 논문과 똑같이 average pooling, tanh를 활성 함수로 사용. <br/>
  conv1 -> 입력 채널 : 1 <br/>
           출력 채널 : 6 <br/>
           커널 사이즈 : 5 x 5 <br/>
           가중치 + 바이어스 : 6 <br/>
           파라미터 갯수 : 6x(1x5x5)+6(weights+biases)=156 <br/>
  conv2 -> 입력 채널 : 6 <br/>
           출력 채널 : 16 <br/>
           커널 사이즈 : 5 x 5 <br/>
           가중치 + 바이어스 : 16 <br/>
           파라미터 갯수 : 16x(6x5x5)+16(weights+biases)=2416 <br/>
  fc1   -> 입력 사이즈 : 16 x 5 x 5 <br/>
           출력 사이즈 : 120 <br/>
           가중치 + 바이어스 : 120 <br/>
           파라미터 갯수 : 120x(16x5x5)+120(weights+biases)=48120 <br/>
  fc2   -> 입력 사이즈 : 120 <br/>
           출력 사이즈 : 84 <br/>
           가중치 + 바이어스 : 84 <br/>
           파라미터 갯수 : 84x120+84(weights+biases)=10164 <br/>
  fc3   -> 입력 사이즈 : 84 <br/>
           출력 사이즈 : 10 <br/>
           가중치 + 바이어스 : 10 <br/>
           파라미터 갯수 : 10x84+10(weights+biases)=850 <br/>
  총 파라미터 갯수 : 61,716 <br/>
  CustomMLP : <br/> 
  fc1   -> 입력 사이즈 : 784 <br/>
           출력 사이즈 : 76 <br/>
           파라미터 갯수 : 76x784+64(weights+biases)=59660 <br/>
 fc2   -> 입력 사이즈 : 76 <br/>
           출력 사이즈 : 10 <br/>
           파라미터 갯수 : 10x76+10(weights+biases)=770 <br/>
  총 파라미터 갯수 : 60,430 <br/>
 <br/>
 <br/>
2. train, test data에 대한 손실 및 정확도 곡선 <br/>
   LeNet5: <br/>
   ![LeNet5](https://github.com/ChangsunLeee/HW2/assets/167077784/ba682db5-463e-4012-907c-b65c6a9f813f) <br/>
   CustomMLP: <br/>
   ![CustomMLP](https://github.com/ChangsunLeee/HW2/assets/167077784/400e9b37-d365-4eb2-bdae-07e63625d4e0) <br/>
 <br/>
 <br/>
4. LeNet5와 CustomMLP의 예측성능 비교: <br/>
   LeNet5의 성능은 98.8%(25분, 100epoch)로 알려져 있음. 노트북의 사양이 그렇게 뛰어나지 않아 10epoch만 돌려보았지만,10epoch의 결과도 평균 0.9879로 유사한 성능을 보임. <br/>
   평균 Train Loss     - 0.06967 <br/>
   평균 Train Accuracy - 0.99025 <br/>
   평균 Test Loss      - 0.04018 <br/>
   평균 Test Accuracy  - 0.98802 <br/>

   CustomMLP의 성능은 10epoch 평균 0.9735%로 0.0145 낮게 나왔음. 대체적으로 LeNet5에 비해 Loss는 높고, Accracy는 낮게나옴. <br/>
   평균 Train Loss     - 0.09205(+0.02238) <br/>
   평균 Train Accuracy - 0.97924(-0.01101) <br/>
   평균 Test Loss      - 0.08966(+0.04948) <br/>
   평균 Test Accuracy  - 0.97350(-0.01452) <br/>
   (출처: https://limepencil.tistory.com/4) <br/>
    <br/>
    <br/>
6. 정규화 기법을 이용한 LeNet5 개선. <br/>
   Standardization -각 특성에 대해 평균을 뺴고 표준 편차로 나누어 특성의 평균이 0, 분산이 1이 되도록 만듬. <br/>
                   -정규 분포로 변환하여 이상치에 덜 민감하게 만듬 <br/>
   Regularization  -모델의 복잡성을 줄이고 가적함을 방지함. <br/>
                   -Lasso(절대값을 패널티)와 Ridge(모델 파라미터의 제곱을 패널티)등이 있음 <br/>
   Scaling         -데이터 범위를 조정하여 다른 특성과 동등한 가중치를 갖도록 만듬. <br/>
                   -Min-Max 스케일링, Z-score 표준화가 사용됨 <br/>
   Normalization   -평균과 분산으로 정규화하여 학습을 안정화하고 더 빠르게 수렴함. <br/>
   Dropout         -신경망의 과적합을 줄이기 위해 사용. 학습 중에 무작위로 선택된 뉴런의 일부를 제외하고 신경망의 연결을 끊는 것. 이를 통해 모델이 특정 뉴런에 과도하게 의존하는 것을 방지하고, 일반화 성능을 향상. <br/>
   이 외에도 많은 방법들이 있지만 Batch Normalization과 Dropout을 이용하여 코드를 개선. <br/>
        <br/>
  ![LeNet5_Upgrade](https://github.com/ChangsunLeee/HW2/assets/167077784/1830fe4d-ae9c-473b-bde9-6d54b5aa7f14)

   평균 Train Loss     - 0.0593(-0.01037) <br/>
   평균 Train Accuracy - 0.9846(-0.00565) <br/>
   평균 Test Loss      - 0.0389(-0.00128) <br/>
   평균 Test Accuracy  - 0.9895(+0.00148) <br/>
   정확도를 조금 잃어버리더라도 loss와의 간격을 줄이는게 중요하다고 하셨는데, 그 간격이 조금 줄어들었음. 오히려 test 데이터 정확도는 올라감. 하지만 적은 훈련수로 인해 조금은 랜덤하게 결과가 나오지 않을까 생각됨.
