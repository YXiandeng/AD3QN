import subprocess
import time


def run_python_file():
    start_time = time.time()
    process = subprocess.Popen(["python", "main_improve_D3QN.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    end_time = time.time()

    print("输出：", stdout.decode())
    print("错误信息：", stderr.decode())
    print("运行时间：", end_time - start_time, "秒")


if __name__ == "__main__":
    num_repeats = 10

    for i in range(num_repeats):
        print(f"第 {i + 1} 次运行...")
        run_python_file()
        if i < num_repeats - 1:
            print("等待5秒后再次运行...")
            time.sleep(5)

    print("所有运行结束！")
