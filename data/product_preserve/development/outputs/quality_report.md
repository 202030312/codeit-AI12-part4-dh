# 기능 3 평가 12건 자동 검사

- 기능 버전: `image_ad_v0.4.1`
- 자동 검사 통과: `4/12`
- 상품 보존 검사는 투명 상품의 불투명 픽셀 색상과 최종 이미지의 동일 위치를 비교한다.
- 문구 안전 영역 검사는 실제 상품 영역과 템플릿의 문구 영역이 겹치지 않는지 확인한다.
- 이전 빈 카드의 경계 대비를 측정해 흰색·검은색 카드가 남지 않았는지 확인한다.
- 최종 육안 검사는 `contact_sheet.png`에서 별도로 수행한다.

| 사례 | 방식 | 템플릿 | 상품 영역 | 색상 차이 | 결과 |
|---|---|---|---|---:|---|
| drink_studio | request_only | - | [451, 151, 1000, 929] | 0.2781 | PASS |
| drink_busy | template_only | fresh_mint | [575, 151, 1000, 929] | 0.6196 | CHECK |
| drink_lowlight | request_and_template | premium_dark | [216, 322, 864, 758] | 0.1310 | CHECK |
| drink_window | request_only | - | [205, 216, 573, 864] | 0.2947 | PASS |
| dessert_studio | template_only | minimal_cream | [216, 267, 864, 813] | 0.3027 | CHECK |
| dessert_busy | request_and_template | pastel_pink | [222, 312, 1000, 768] | 0.5108 | CHECK |
| dessert_lowlight | request_only | - | [291, 355, 788, 724] | 0.1367 | PASS |
| dessert_window | template_only | pastel_pink | [140, 310, 788, 769] | 0.3242 | CHECK |
| bread_studio | request_and_template | clean_white | [151, 219, 929, 861] | 0.3319 | CHECK |
| bread_busy | request_only | - | [80, 287, 858, 792] | 0.4615 | PASS |
| bread_lowlight | template_only | premium_dark | [216, 290, 864, 790] | 0.1904 | CHECK |
| bread_window | request_and_template | natural_beige | [292, 288, 940, 792] | 0.2300 | CHECK |
