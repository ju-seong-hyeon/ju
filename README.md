# 머신러닝 기반 도면 객체인식 알고리즘 개발 및 추출
머신러닝 기반 도면 객체인식 알고리즘 개발 및 추출 Private  Updated 2 minutes ago 소방시설 정보가 전산화되어 있지 않고 이미지(CAD)도면 형태로만 남아있어 모든 소방시설 정보를 수작업으로 입력하고 관리하여 비효율을 초래한다. 또, 소수의 인원으로 모든 건물의 소방시설을 관리하기에 어려움이 있어 인명, 재산 피해가 나타난다. 이를 극복하기 위해 도면상의 소방시설정보를 머신러닝으로 인식하고, xml파일로 저장하여 다른 플랫폼과 연계할 수 있도록 한다.  도면에 있는 소방시설을 인식하기 위해 도면과 소방시설기호를 전처리를 하였으며 maskrcnn으로 인식한 후 인식된 소방시설기호의 정보를 xml파일로 저장한다.
