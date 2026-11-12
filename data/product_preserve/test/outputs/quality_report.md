# 기능 3 평가 12건 자동 검사

- 기능 버전: `image_ad_v0.4.1`
- 자동 검사 통과: `8/12`
- 상품 보존 검사는 투명 상품의 불투명 픽셀 색상과 최종 이미지의 동일 위치를 비교한다.
- 문구 안전 영역 검사는 실제 상품 영역과 템플릿의 문구 영역이 겹치지 않는지 확인한다.
- 이전 빈 카드의 경계 대비를 측정해 흰색·검은색 카드가 남지 않았는지 확인한다.
- 최종 육안 검사는 `contact_sheet.png`에서 별도로 수행한다.

| 사례 | 방식 | 템플릿 | 상품 영역 | 색상 차이 | 결과 |
|---|---|---|---|---:|---|
| test_bread_cream_brioche | template_only | cream | [130, 170, 950, 930] | 0.1815 | PASS |
| test_bread_olive | template_only | natural | [55, 428, 605, 742] | 0.0902 | PASS |
| test_bread_red_bean | template_only | berry | [125, 434, 955, 935] | 0.3520 | CHECK |
| test_bread_salt | template_only | promo | [125, 121, 955, 659] | 0.2201 | PASS |
| test_dessert_fruit_tart | template_only | berry | [130, 234, 950, 895] | 0.4333 | PASS |
| test_dessert_mango_bingsu | template_only | promo | [475, 326, 1025, 843] | 0.0430 | CHECK |
| test_dessert_mont_blanc | template_only | night | [55, 276, 605, 894] | 0.1196 | PASS |
| test_dessert_pavlova | template_only | cream | [125, 351, 955, 1019] | 0.0292 | PASS |
| test_drink_black_sesame | template_only | night | [475, 388, 1025, 782] | 0.3091 | CHECK |
| test_drink_cappuccino | template_only | clean | [130, 269, 950, 861] | 0.1397 | CHECK |
| test_drink_earl_grey | template_only | natural | [301, 55, 779, 725] | 0.0469 | PASS |
| test_drink_watermelon | template_only | clean | [277, 105, 803, 1024] | 0.4567 | PASS |
