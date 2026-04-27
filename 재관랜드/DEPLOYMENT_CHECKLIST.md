# K-Quant Deck 배포 체크리스트

이 문서는 다른 PC나 포트폴리오용 ZIP으로 프로젝트를 옮길 때 민감정보가 섞이지 않도록 확인하는 기준입니다.

## 포함해도 되는 파일

- `kis_autotrade.py`
- `web_dashboard.py`
- `templates/`
- `static/`
- `README.md`
- `.env.example`
- `run_k_quant_deck.bat`
- `start_dashboard.bat`
- `stop_k_quant_deck.bat`
- `install_dependencies.bat`
- `make_release.ps1`
- 학습/검증용 Python 스크립트

## 절대 포함하면 안 되는 파일

- `.env`
- `흠흠.env`
- `흠흠.txt`
- `.cache/`
- `kis_token_real.json`
- 실제 계좌번호, API 키, API 시크릿, 접근 토큰이 들어간 모든 파일

## 새 PC에서 실행하는 순서

1. Python을 설치합니다.
2. `install_dependencies.bat`을 한 번 실행합니다.
3. `.env.example`을 복사해서 `.env` 파일을 만듭니다.
4. `.env` 안에 본인 한국투자 Open API 키, 시크릿, 계좌번호를 넣습니다.
5. `run_k_quant_deck.bat`을 실행합니다.
6. 브라우저에서 `http://127.0.0.1:5050`으로 접속합니다.

## 배포 전 최종 확인

- ZIP 안에 `.env`가 없는지 확인합니다.
- ZIP 안에 `.cache` 폴더가 없는지 확인합니다.
- README에 로컬 절대경로와 실제 계좌번호가 없는지 확인합니다.
- 실거래 기본값이 의도한 설정인지 확인합니다.
