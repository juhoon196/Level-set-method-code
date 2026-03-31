#!/bin/bash
#SBATCH --job-name=test_A
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48           # 노드 하나를 통째로 쓰기 위해 48 설정
#SBATCH --output=test_A.out          # 표준 출력 저장
#SBATCH --error=test_A.err           # 에러 로그 저장

# [중요] 모듈 로드 확인
# 만약 아까 '모듈이 없는 느낌'이었다면, 여기에 컴파일할 때 썼던 모듈을 로드해야 합니다.
# 예: module load gcc (정확한 이름은 module avail로 확인)

# 현재 디렉토리 정보 출력 (디버깅용)
echo "Working directory: $(pwd)"
echo "Host: $(hostname)"

# 프로그램 실행 (주파수 10Hz 테스트)
./gsolver2d_A 10
