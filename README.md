GPU서버 스트레스 테스트용 코드입니다. 3090x4 서버 기준 약 4일 동안 진행됩니다. 스트레스 테스트 진행 시간은 `stress_test.sh`에서 max_epochs 값을 통해 조절이 가능합니다.


## 실행 방법

### install.sh 실행
`install.sh`를 실행하여 스트레스 테스트 진행을 위한 필요한 패키지 설치 및 환경을 설정합니다. Anaconda3가 설치되며 필요한 파이썬 패키지도 함께 설치됩니다.
Anaconda3 쉘 환경을 위해 source로 `install.sh`를 실행시킵니다.
``` 
source install.sh
```


### stress_test.sh 실행
스트레스 테스트를 진행합니다. 

```
bash stress_test.sh
```
