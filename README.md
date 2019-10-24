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
- 밀도 기반 기법 (density-based method): 분할 기법이 거리에 기초하여 클러스터를 만드는 것과 달리 밀도 개념에 기초하여 발전해 왔다. 밀도 기반 기법은 한 데이터 포인트를 기준으로 일정 반경 내에 기준 이상의 데이터 포인트가 포함되면 클러스터를 구성하는 방식이다. 데이터 밀도가 높은 부분은 클러스터가 형성되고 데이터 밀도가 낮은 부분은 클러스터의 경계가 형성되며 경계 밖의 데이터는 이상치로 간주된다. 밀도 기반 클러스터링 알고리즘은 DBSCAN과 OPTICS 등이 있다. Chandrakala (2008)은 kernel feature 공간에 밀도 기반 클러스터링 알고리즘을 사용하여 길이가 다른 다변량 시계열 데이터를 클러스터링하는 방법을 제안했다.
- 그래프 기반 기법 (graph-based method): 그래프 기반 기법은 그래프 이론에 따라 데이터를 노드로 구성하고 관측치 간 거리를 가중치로 정의한다. 데이터를 그래프 화 한 후 가중치가 낮은 간선을 끊어 클러스터링 한다.

<p align="center">
  <img width="600" src="/assets/spectral clustering.png">
</p>

- 격자 기반 기법 (grid-based method): to be updated
- 모델 기반 기법 (model-based method): to be updated
<!-- ## Table of Contents
- [Time-series anomaly detection](#time-series-anomaly-detection)
- [Image-level anomaly detection](#image-level-anomaly-detection)
  - [Classification target](#classification-target)
  - [Segmentation target](#segmenatation-target) -->

## Survey Paper 
- S. Aghabozorgi, A.S. Shirkhorshidi, T.Y. Wah, (2015). Time-series clustering - A decade review| **Information Systems** | [`[pdf]`](https://reader.elsevier.com/reader/sd/pii/S0306437915000733?token=4BF3F6164FB27C86C6256D98B2E62DEF3EE5D7E94DC5391015D059DE5043C4D13DAB714030F3F6FA703537E2C3CD1DC9)
- TW Liao, (2005). Clustering of time series data - a survey. 
<p><i>Pattern Recognition</i></p>
| [`[pdf]`](https://reader.elsevier.com/reader/sd/pii/S0031320305001305?token=1DAAE29A44F329438C9671499FE2ECE77A807C19B77B0649D836304757F42A9B0EED63E54A0058FCA193E8DBE315EDB8)

## Clustering algorithm & papers

### Density-based clustering
- M. Ester, H.P. Kriegel, J. Sander, X. Xu, (1996). A density-based algorithm for discovering clusters in large spatial databases with noise 
<p><i>KDD' 96</i></p>
| [`[pdf]`](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)


<!-- 
## Contact & Feedback
If you have any suggenstions about papers, feel free to mail me :)
- [e-mail](mailto:lee.hoseong@sualab.com)
- [blog](https://hoya012.github.io/)
- [pull request](https://github.com/hoya012/awesome-anomaly-detection/pulls) -->