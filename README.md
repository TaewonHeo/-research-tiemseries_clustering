# Timeseries Clustering
A curated list of timeseries clustering resources. Inspired by [`awesome-anomaly-detection`](https://github.com/hoya012/awesome-anomaly-detection)).

*Last updated: 2019/10/24*

## What is timeseries clustering?

<p align="center">
  <img width="300" src="/assets/clustering.jpg">
</p>

시계열 클러스터링은 레이블이 없는 시계열 데이터를 이용하여 서로 패턴이 유사한 시계열 데이터들을 적절한 그룹들로 구분하는 클러스터를 형성하는 것이 목적이다. 본 연구에서는 시계열 클러스터링을 이용해 새로운 서비스자재의 수요패턴을 예측하는 과정에서 기존의 데이터베이스에 축적된 데이터 중 모델 학습에 유의미한 데이터를 선별하여 머신러닝 / 딥러닝 모델의 학습 효과를 극대화할 예정이다. 시계열 클러스터링 과정에서 데이터의 가공 방식(연속형 여부, 다변량 여부, 시계열의 동기화 및 길이의 동일성 여부), 데이터 간의 거리 측정 방식, 클러스터링 방식에 따라 클러스터링의 결과가 달라질 수 있다.
클러스터링 기법은 크게 분할 기법, 계층적 기법, 밀도 기반 기법, 격자 기반 기법, 모델 기반 기법으로 나눌 수 있다. 그 중 시계열 데이터의 클러스터링에 많이 쓰이는 기법은 다음과 같다.
 

## Clustering method
클러스터링 기법은 크게 분할 기법, 계층적 기법, 밀도 기반 기법, 격자 기반 기법, 모델 기반 기법으로 나눌 수 있다. 그 중 시계열 데이터의 클러스터링에 많이 쓰이는 기법은 다음과 같다.

- 분할 기법 (partitioning methods): k개의 초기 클러스터를 만든 후 각각의 객체를 하나의 클러스터에서 다른 그룹으로 재배치하는 작업을 반복함으로써 클러스터를 만들어나간다. 대표적인 알고리즘은 k-means 알고리즘으로 간단하고 빠르지만, 임의로 정해지는 초기값에 매우 민감하여 결과가 매번 달라질 수 있다는 것이 단점이다. 이러한 단점을 극복하기 위해, 초기 클러스터의 중심을 계층적 클러스터링 방법으로 최적화한 후 k-means 알고리즘을 사용하는 modified k-means 알고리즘이 개발되었다.  
- 계층적 기법 (hierarchical method): 순차적으로 가까운 관측 값끼리 묶어주는 병합 방법과 먼 관측 값들을 나누어 가는 분할 방법이 있으며, 주로 병합 방법을 사용하여 클러스터를 bottom-up 방식으로 병합한다. 계층적 기법은 시계열 데이터를 클러스터링 결과를 시각화 할 수 있다는 점에서 강력하다. 또한, 길이가 다른 시계열 데이터도 거리 측도로 Dynamic Time Warping (DTW)를 사용하여 클러스터링이 가능하다는 장점이 있다. 
Average Linkage Agglomerative Clustering 알고리즘은 계층적 기법 중 bottom-up 방식의 병합 방식의 알고리즘이다. 계층적 클러스터링은 병합할 때 사용하는 거리 계산 방법(연결 방법)에 따라 상이한 결과가 도출된다. 본 연구에서 사용할 연결 방법은 평균 연결법 (average linkage)이며, 클러스터 u 의 모든 데이터 i 와 클러스터 v 의 모든 데이터 j 의 모든 조합에 대해 거리를 측정한 후 평균을 구하는 식은 아래와 같다.
d(u,v)=∑_(i,j)▒(d(u[i],v[j]))/(|u||v|)
위의 평균 연결법을 이용하여 가장 가까운 클러스터를 병합해가며 다음과 같이 계층적 클러스터링 절차를 수행한다.
	전체 데이터를 개별 클러스터로 취급한다. 개별 관측치를 C_1,C_(2,)⋯〖,C〗_n 이라 했을 때, i번째 데이터와 j번째 데이터 사이의 거리를 d_(C_i C_j )=d_ij라고 한다.
	가장 작은 거리 측도 값을 가지는 두 개의 클러스터를 찾고, 이를 C_K 와 C_M 이라고 한다.
	C_K 와 C_M을 통합하여 하나의 클러스터로 취급하고 이를 C_KM이라고 한다. C_K 와 C_M은 클러스터링 되었으므로 개별적 통합의 대상에서 제외한다.
	새롭게 생성된 군집 C_KM과 나머지 모든 클러스터와의 거리를 평균 연결법으로 다시 계산한다. 
2번부터 4번까지의 절차를 클러스터의 개수가 지정한 개수가 될 때까지 반복한다

- 밀도 기반 기법 (density-based method): 분할 기법이 거리에 기초하여 클러스터를 만드는 것과 달리 밀도 개념에 기초하여 발전해 왔다. 밀도 기반 기법은 한 데이터 포인트를 기준으로 일정 반경 내에 기준 이상의 데이터 포인트가 포함되면 클러스터를 구성하는 방식이다. 데이터 밀도가 높은 부분은 클러스터가 형성되고 데이터 밀도가 낮은 부분은 클러스터의 경계가 형성되며 경계 밖의 데이터는 이상치로 간주된다. 밀도 기반 클러스터링 알고리즘은 DBSCAN과 OPTICS 등이 있다. Chandrakala (2008)은 kernel feature 공간에 밀도 기반 클러스터링 알고리즘을 사용하여 길이가 다른 다변량 시계열 데이터를 클러스터링하는 방법을 제안했다. 밀도 기반 클러스터링 기법 중 하나인 DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 알고리즘은 데이터 공간에서 밀도가 높은 부분을 클러스터링 하는 방식으로, 한 점을 기준으로 반경 Epsilon (Eps) 내에 점이 m개 이상 있으면 군집으로 인식한다. 모델 파라미터인 minimum number of points in Eps-neighborhood of that point (MinPts)는 클러스터를 구성하기 위해 Eps 반경 내에 포함 되어야하는 최소한의 이웃 수를 나타낸다. N_Eps (p)는 아래와 같이 정의된다.
N_Eps (p)={q∈D|dist(p,q)≤Eps)
한 클러스터의 내의 점들은 MinPts 기준을 넘는지에 따라 중심점 (core point)과 경계점(border point)으로 나뉜다. MinPts를 4라고 할 때 아래의, 점 P는 epsilon 반경 안에 5개의 점이 있으므로 중심점의 조건을 만족한다. 
<p align="center">
  <img width="700" src="/assets/dbscan1.jpg">
</p>          
반면, 우측의 점 P2는 P의 기준에서는 클러스터 내에 위치하지만 P2를 기준으로 이웃하는 점의 수가 MinPts를 만족하지 못하므로 경계점이 되어 클러스터의 경계를 구성하게된다.
<p align="center">
  <img width="700" src="/assets/dbscan2.jpg">
</p>       
좌측의 P3와 P는 같은 반경 내에서 모두 중심점이 되므로 P와 P3는 같은 클러스터로 묶인다. 우측의 P4는 반경 내에 어떤 중심점도 포함되지 않으므로 이상치로 간주한다. DBSCAN은 길이가 다른 다변량 시계열 데이터의 클러스터링에 적용할 수 있다는 점에서 이점이 있고, 군집의 수를 결정하지 않아도 된다는 점에서 시계열 데이터의 클러스터링에 적절한 알고리즘이다.

- 그래프 기반 기법 (graph-based method): 그래프 기반 기법은 그래프 이론에 따라 데이터를 노드로 구성하고 관측치 간 거리를 가중치로 정의한다. 데이터를 그래프 화 한 후 가중치가 낮은 간선을 끊어 클러스터링 한다.

<p align="center">
  <img width="600" src="/assets/spectral clustering.png">
</p>
Spectral clustering은 그래프 기반의 클러스터링 알고리즘으로 길이가 다른 시계열 데이터의 클러스터링에 적용 가능한 알고리즘이다. 그래프 구축을 위해 데이터 간 거리를 인접 행렬 (adjacency matrix)로 나타내고, 행렬 계산 시 가까운 데이터에 대해 높은 유사도를 주기 위해 가우시안 커널을 이용한다. 그래프 구축 후 그래프를 cut 규칙에 의해 분할하여 클러스터로 만든다. 그래프 분할 시, 원 그래프를 A, B의 두 부 그래프로 나눌 때 끊어지는 간선의 가중치가 최소화되도록 하는 목적함수는 아래와 같다. 
MinCut(A,B)=min 1/4 q^t (D-W)q
여기서 W는 가중치 행렬이고 대각 행렬 D와 벡터 q는 아래와 같다. 
D_ii=∑_j▒w_ij ,D_ij=0 if i≠j
q_i= {■(1  if i∈A@-1  if i∈B)}
이 방법은 하나의 그래프를 2개의 부 그래프로 나누는 방법이며 여러 개의 부 그래프로 클러스터링하기 위해서 위의 방법을 반복수행한다. 

- 격자 기반 기법 (grid-based method): to be updated
- 모델 기반 기법 (model-based method): to be updated
<!-- ## Table of Contents
- [Time-series anomaly detection](#time-series-anomaly-detection)
- [Image-level anomaly detection](#image-level-anomaly-detection)
  - [Classification target](#classification-target)
  - [Segmentation target](#segmenatation-target) -->

## Survey Paper 
- Time-series clustering - A decade review | S. Aghabozorgi, A.S. Shirkhorshidi, T.Y. Wah, (2015) | **Information Systems** | [`[pdf]`](https://reader.elsevier.com/reader/sd/pii/S0306437915000733?token=4BF3F6164FB27C86C6256D98B2E62DEF3EE5D7E94DC5391015D059DE5043C4D13DAB714030F3F6FA703537E2C3CD1DC9)
- Clustering of time series data - a survey | TW Liao, (2005) | **Pattern Recognition** | [`[pdf]`](https://reader.elsevier.com/reader/sd/pii/S0031320305001305?token=1DAAE29A44F329438C9671499FE2ECE77A807C19B77B0649D836304757F42A9B0EED63E54A0058FCA193E8DBE315EDB8)

## Clustering algorithm & papers

### Density-based clustering
- A density-based algorithm for discovering clusters in large spatial databases with noise | M. Ester, H.P. Kriegel, J. Sander, X. Xu, (1996) | **KDD' 96** | [`[pdf]`](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)


<!-- 
## Contact & Feedback
If you have any suggenstions about papers, feel free to mail me :)
- [e-mail](mailto:lee.hoseong@sualab.com)
- [blog](https://hoya012.github.io/)
- [pull request](https://github.com/hoya012/awesome-anomaly-detection/pulls) -->