# Measuring_angle

## 이미지 프로세싱을 이용한 교량 하단을 촬영하는 드론의 헤딩 계산

* 기간 : 19 겨울학기
* 목표 : 이미지 프로세싱을 이용하여 교량 하단 영상에서 기울어진 각도 획득
* 사용언어/프로그램 : python, OpenCV

### 프로그램
```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```
영상의 이미지를 그레이 스케일로 변환해줌
```
bilateral_img=cv2.bilateralFilter(gray,3,20,20)
```
저력통과필터를 통해 노이즈 제거 시 상세 정보가 손상되어 검출되는 에지의 위치의 정확도가 떨어짐.
양방향 필터(bilateral filter)를 이용해 잡음을 제거하고 경계선을 뚜렸하게 해줌.

[OpenCV 메뉴얼](https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=bilateral#cv2.bilateralFilter)

- d - 필터링 중에 사용되는 각 픽셀 근처의 직경. 양성이면, sigmaSpace로부터 계산된다.

- sigmaColor는 - 색 공간에서 시그마를 필터링합니다. 상기 파라미터의 값이 클수록 화소 근방 (sigmaSpace 참조) 내에서 더 멀리 색 세미 동일한 색상의 큰 영역의 결과를 함께 혼합 될 것이라는 것을 의미한다.

- sigmaSpace는 - 좌표 공간에서 시그마를 필터링합니다. 매개 변수의 값이 클수록 더 멀리 픽셀만큼 자신의 색 (sigmaColor 참조) 충분히 가까이 서로 영향을 미치는 것을 의미합니다. d> 0 거라고 때없이 sigmaSpace의 근방 크기를 지정한다. 그렇지 않으면, d를 sigmaSpace에 비례한다.

```python
clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(4,4))
CLAHE_img = clahe.apply(bilateral_img)
CannyThresh = CannyAccThresh/4*3
```
이미지의 히스토그램이 한쪽으로 치우쳐져 있으면 한쪽이 너무 밝아 상대적으로 어두운 현상이 발생하므로 히스토그램을 펼쳐주는 평활화 작업이 필요함.

```python
CannyAccThresh = cv2.threshold(CLAHE_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
CannyAccThresh=(CannyAccThresh+20)
CannyThresh = CannyAccThresh/4*3
```
이미지의 밝기에 따라 canny 에지 검출알고리즘의 임계값을 자동으로 계산해주어야 함. 임계값은 ...을통해주며 ..값과 임계값의 비는 실험적으로 구했음

```python
edges = cv2.Canny(CLAHE_img, CannyThresh,CannyAccThresh)
```
canny를 통해 엣지를 얻음.

[OpenCV 메뉴얼]()
  - 쓰레시 홀드 필터가 이동하면서 픽셀의 그레디언트high값보다 크면 엣지, low보다 작으면 엣지가 아니라고 인식. low와 high 사이에 있으면 그 주위에 엣지가 있 지 확인 후에 있으 엣지라고 인식

```python
mask = np.zeros_like(edges)
ignore_mask_color = 255
vertices = np.array([[(int(w/2)-gap,h),(int(w/2)-gap, 0), (int(w/2)+gap,0),(int(w/2)+gap,h)]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)
```
Ha...를 통해 에지 검출시 소요시간이 기므로 ROI()를 지정하며 시간을 줄여줌. 교량의 하단은 가로선이 두드러지므로 아래와 같이 ROI를 설정함.



### 정확도
| <center>표준편차 </center> |  해 방향 |  역방향 |
|:--------|:--------:|--------:|
| <center>걷기 </center> | <center>0.7540 </center> |1.9140 |
| <center>정지 </center> | <center>0.7008</center> | 0.1178 |

카메라를 사람이 들고 영상을 촬영해 흔들림에의해 표준편차가 발생하였음.

걷는 영상의 길아는 16초, 정지 영상의 길이는 61초임.

교량의 외각선이 지저분한 경우(사진 상 곡선이거나, 여러개의 선이 겹쳐 보이는 경우) 정확도가 떨어진다. 하지만 어떤 경우에도 강인한 프로그램이 필요함.
