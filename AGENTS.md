# Autonomous Loop

## Goal

`A_Joint_Learning_and_Communications_Framework_for_Federated_Learning_Over_Wireless_Networks.pdf` 논문의 Fig. 3-10은 검증 기준이며, 구현 목표는 그림을 흉내 내는 것이 아니라 논문에 기술된 federated learning over wireless networks 실험을 실제로 재현하는 것이다.
Figures are validators, not the implementation target.

## Rule

1. 설계, 구현, 답변을 하기 전 A_Joint_Learning_and_Communications_Framework_for_Federated_Learning_Over_Wireless_Networks.pdf 논문을 읽는다.
2. 구현 시 blueprints/* 에 정의된 파일과 함수 외 추가적으로 다른 전역변수, 함수, 파일을 생성하지 않는다. 만약 추가가 불가피하다면 사용자와 토론을 진행한 후 승낙을 받고 추가한다.
3. outputs 폴더 아래에 결과 그래프들을 저장한다. 결과 그래프의 스타일은 논문과 동일하게 한다.
4. git으로 형상을 관리한다.
5. 논문에 기재되지 않아 스스로 찾아야 하는 파라미터는 configs 폴더 아래 yaml 파일로 정의하고 sweep할 수 있어야 한다.

## BAN

1. smoke test를 금지한다.