import os
import glob
import platform
import shutil
import timeit

from decimal import Decimal
from datetime import datetime
from collections import Counter, defaultdict


# 시간 측정
# 경로 분리
# 시스템 정보, 경로, 디렉토리
# 파일 목록 조회
# 데이터 형변환
# 문자열
# 디렉토리 생성 복사 삭제 이동
# 자료형(list, dict) 값 확인
# decimal 사칙연산
# 소수점 조정
# floatRange

#------------------------------------------------------------------------------------------
# 시간 측정
def getTimer():
    """
    * 측정 포인트의 시간 객체 반환
    :return: timer 객체
    """
    return timeit.default_timer()

def getRunningTime(startTimer, stopTimer, runningName):
    """
    * 소요되는 시간을 출력
    :param startTimer: 시작 시점 타이머
    :param stopTimer:  종료 시점 타이머
    :param object: 측정된 포인트 이름
    :return:
    """
    #print("--->>> Running Time {} sec - '{}'".format(round(stopTimer-startTimer, 3), runningName))

    seconds = round(stopTimer-startTimer, 3)
    seconds = int(seconds)
    #status = 'has been running for' if not finished else 'finished in'

    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    periods = [('시간', hours), ('분', minutes), ('초', seconds)]
    time_string = ', '.join('{} {}'.format(value, name)
                            for name, value in periods
                            if value)

    #print('The script {} {}'.format(status, time_string))



    return "--->>> Running Time {} - '{}'".format(time_string, runningName)

# 시간 차이 
def getDeltaTime(date1, date2, timeFormat = "%H:%M"):
    """
    * 입력한 두 시간의 차이 
    :param date1: 이후 시간 문자열
    :param date2: 이전 시간 문자열
    :param timeFormat: 시간 포멧 형식 문자열
    :return: 시간차
    """
    time1 = datetime.strptime(date1, timeFormat)
    time2 = datetime.strptime(date2, timeFormat)
    delta = time1 - time2
    return delta
#------------------------------------------------------------------------------------------
# 경로 분리

def getFileName(filePath):
    """
    * 확장자 포함 파일명 추출
    :param filePath: 파일 전체 경로
    :return:  확장자 포함 파일명
    """
    return os.path.basename(filePath)
#print(getFileName("E:/TF_Work/Excel/PersonDB.xlsx"))

def getOnlyFileName(filePath):
    """
    * 확장자를 제외한 파일명 추출
    :param filePath: 파일 전체 경로
    :return: 확장자 제외한 파일명
    """
    tokens = os.path.basename(filePath).split(".")
    return tokens[0]
#print(getOnlyFileName("E:/TF_Work/Excel/PersonDB.xlsx"))

def getFileExtension(filePath):
    """
    * 파일 확장자 추출
    :param filePath: 파일 전체 경로
    :return: 파일의 확장자
    """
    return os.path.basename(filePath.split(".")[-1])
#print(getFileExtension("E:/TF_Work/Excel/PersonDB.xlsx"))

def getParentDirPath(filePath):
    """
    * 입력한 파일의 부모 디렉토리 경로
    :param filePath: 파일 전체 경로
    :return: 부모 디렉토리 경로
    """
    return os.path.dirname(filePath)
#print(getParentDirPath("E:/TF_Work/Excel/PersonDB.xlsx"))

def isDir(filePath):
    """
    * 해당 경로가 파일인지 폴더인지 확인
    :param filePath: 
    :return: True / False
    """
    return os.path.isdir(filePath)

def getAllDirElementList(filePath):
    """
    * Directory path만 반환
    :param filePath: 
    :return: dir path List
    """
    result = list()
    fileNames = os.listdir(filePath)
    for fileName in fileNames:
        fullPath = os.path.join(filePath, fileName)
        if os.path.isdir(fullPath):
            result.append(fullPath)
    return result


#------------------------------------------------------------------------------------------
# 시스템 정보, 경로, 디렉토리
def getTimeLabel():
    """
    * 현재 시간을 문자열로 반환함
    :return: 연월일_시분초 형태의 현재 시간을 문자열로 반환
    """
    now = datetime.now()
    label = "{0}{1:0>2}{2:0>2}_{3:0>2}{4:0>2}{5:0>2}"
    #label = "{0}-{1:0>2}-{2:0>2} {3:0>2}:{4:0>2}:{5:0>2}"
    return label.format(now.year, now.month, now.day, now.hour ,now.minute, now.second)


def getSystemType() :
    """
    * 시스템 OS 종류 확인
    :return:
    """
    return platform.system()
#print (getSystemType())

def getSep():
    """
    * OS에 따른 디렉토리 분리 문자 반환
    :return: OS 별 디렉토리 분리문자
    """
    return os.sep
#print (getSep())

def changeSeperator(path):
    """
    * 경로의 File Seperator 를 OS 에 맞게 변경
    :param path: 변경 할 경로
    :return: 변경된 경로
    """
    return path.replace("\\", os.sep).replace("/", os.sep)
#print(changeSeperator("E:\\TF_Work\Excel/JSLib.py"))

def getCurDir():
    """
    * 현재 프로세스의 작업 디렉토리 경로 확인
    :return: 현재 작업 디렉토리 경로
    """
    return os.getcwd()
#print(getCurDir(), "  -  " , type(getCurDir()))


def getNewPath(path, add):
    """
    * 새롭게 경로를 만듬
    :param path: 기본경로
    :param add: 파일명 혹은 뒷부분 경로 추가
    :return: 새롭게 만들어진 경로
    """
    return os.path.join(path, add)
#print(setPath)

def isExist(path):
    """
    * 경로가 존재하는지 확인
    :param path: 경로 입력
    :return: 존재할경우 True, 존재하지 않을 경우 False
    """
    return os.path.exists(path)
#print (isExist("E:\TF_Work"))

def getAbsPath(path):
    """
    상대경로를 절대 경로로 변환
    :param path: 상대 경로
    :return: 절대 경로
    """
    return os.path.abspath(path)

#------------------------------------------------------------------------------------------
# 파일 목록 조회

def getAllElementsPathList(path):
    """
     * 입력한 디렉토리의 모든 항목(파일+디렉토리)의 경로를 리스트로 반환
    :param path: 조회할 디렉토리 전체 경로
    :return: 모든 항목들의 경로 리스트
    """
    result = list()
    fileNames = os.listdir(path)
    for fileName in fileNames:
        fullPath = os.path.join(path, fileName)
        result.append(fullPath)
    return result
#print(getAllElementsPathList("E:\\TF_Work"))

def getAllElementsNameList(path):
    """
    * 입력한 디렉토리의 모든 항목(파일+디렉토리)의 이름을 리스트로 반환
    :param path: 조회할 디렉토리 전체 경로
    :return: 모든 항목들의 이름 리스트
    """
    return os.listdir(path)
#print (getAllElementsNameList("E:\TF_Work\Excel"))

def getElementPathListWithPattern(path , pattern):
    """
    * 입력한 디렉토리의 pattern 을 가진 항목 리스트 반환
    :param path: 조회할 디렉토리 전체 경로
    :param pattern: 조회활 패턴 입력(*, *.*, *.txt 등등)
    :return: 조회된 결과 항목 리스트
    """
    fileType = os.path.join(path,pattern)
    return glob.glob(fileType)
#print(getElementListWithPattern("E:/TF_Work","*"))


#------------------------------------------------------------------------------------------
# 데이터 형변환
def isString(value):
    """
    * str 형 확인
    :param value: 입력 데이터
    :return: 확인 결과 True  / False
    """
    if type(value) is str:
        return True
    else:
        return False

def isInteger(value):
    """
    * Int 형  확인
    :param value: 입력 데이터
    :return: 확인 결과 True / False
    """
    if type(value) is int:
        return True
    else:
        return False


def isFloat(value):
    """
    * float 형 확인
    :param value: 입력데이터
    :return: 확인 결과 True / False
    """
    if type(value) is float:
        return True
    else:
        return False

def isList(value):
    """
    * List 형 확인
    :param value: 입력데이터
    :return: 확인 결과 True / False
    """
    if type(value) is list:
        return True
    else:
        return False

def isDict(value):
    """
    * Dictionary 형 확인
    :param value: 입력데이터
    :return: 확인 결과 True / False
    """
    if type(value) is dict:
        return True
    else:
        return False

def isCharcter(value):
    """
    * 문자인지 확인
    :param value: 문자 or 문자열
    :return: 확인 결과 True | False
    """
    try:
        int(value)
        return False
    except:
        return True




#------------------------------------------------------------------------------------------
# 문자열
def removeSideBlank(value):
    """
    * 문자열의 양쪽 공백 지우기
    :param value: 입력 문자열
    :return: 공백 지운 문자열
    """
    return value.strip()

def removeLeftSideBlank(value):
    """
    * 문자열 왼쪽 공백 지우기
    :param value: 입력 문자열
    :return:
    """
    return value.lstrip()

def removeRightSideBlank(value):
    """
    * 문자열 오른쪽 공백 지우기
    :param value: 입력 문자열
    :return: 공백 지운 문자열
    """
    return value.rstrip()

def removeAllBlank(value):
    """
    * 문자열의 모든 공백을 지움
    :param value: 입력 문자열
    :return: 공백 지운 문자열
    """
    result = list()
    value = value.strip()

    for c in value:
        n_c = c.strip()
        result.append(n_c)
        # print("{} => {}".format(c, n_c))
        # print("c size : {} => n_c size : {}".format(len(c),len(n_c)))
    return "".join(result)



def splitToken(value, delimiter = None):
    """
    * delimiter를 기준으로 문자열 분리(공백, 탭, 개행 문자열은 입력 생략)
    :param value: 입력 문자열
    :param delimiter: 분리 기준 문자
    :return: 분리된 문자열 리스트
    """
    if delimiter is None:
        return value.split()
    else:
        return value.split(delimiter)

#print(splitToken("a, b, c,,e",','))
# ['a', ' b', ' c', '', 'e']


def convertToString(value):
    """
    * 문자열로 변환
    :param value: 입력데이터
    :return: 변환한 데이터
    """
    return repr(value)


def convertToInteger(value):
    """
    * int 형 변환
    :param value: 입력데이터
    :return: 변환한 데이터 (None 이면 형변환 실패)
    """
    if isInteger(value):
        return int(value)
    elif isString(value):
        result = -1
        try:
            result = int(value)
        except:
            return None
        return result
    else:
        return None


def convertToFloat(value):
    """
    * float 형 변환
    :param value: 입력데이터
    :return: 변환한 데이터 (None 이면 형변환 실패)
    """
    if isFloat(value):
        return float(value)
    elif isString(value):
        result = 0
        try:
            result = float(value)
        except:
            return None
        return result
    else:
        return None

def getCSVDataLine(dataList = None):
    """
    * csv 파일로 저장하기 위하여 데이터를 ' , ' 로 연결한 문자열 반환
    :param dataList: 입력데이터 (리스트) 
    :return: , 로 연결된 문자열
    """
    if type(dataList) is list:
        return ','.join(dataList)
    else:
        return None

def format_E_6(value):
    """
    *  E 가 포함된 형태로 소수점 6자리 표현
    :param value: 입력 값
    :return: 변환된 문자열 반환
    """
    if isFloat(value):
        return format(value,"10.6E")
    else:
        return format(convertToFloat(value), "10.6E")


def format_E_7(value):
    """
    * E 가 포함된 형태로 소수점 7자리 표현
    :param value: 입력 값
    :return: 변환된 문자열 반환
    """
    if isFloat(value):
        return format(value, "10.7E")
    else:
        return format(convertToFloat(value), "10.7E")

def strContains(data, findValue):
    """
    * 문자열에 findValue 가 포함되었는지 판단
    :param data: 입력 문자열
    :param findValue: 찾는 문자
    :return: True / False
    """
    index = data.find(findValue)

    if index == -1:
        return False
    else:
        return True

def strListContains(data, findValueList):
    """
    * 문자열에 findValueList 가 포함되었는지 판단
    :param data: 입력 문자열
    :param findValueList: 찾는 문자열 리스트
    :return: True / False
    """
    result = False
    index = 0
    for findVal in findValueList:
        index = data.find(findVal)
        if index == -1:
            pass
        else:
            result = True
            break

    return result

def strUpper(value):
    """
    * 문자열 대문자로 변환
    :param value: 입력데이터 
    :return: 대문자 변환 문자열
    """
    return value.upper()

def strLower(value):
    """
    * 문자열 소문자로 변환
    :param value: 입력데이터 
    :return: 소문자 변환 문자열
    """
    return value.lower()



#------------------------------------------------------------------------------------------
# 디렉토리 생성 복사 삭제 이동
def copyFile(srcPath, dstPath):
    """
    * 파일 복사
    :param srcPath: 원본 파일 
    :param dstPath: 대상 파일
    :return: 없음
    """
    if isExist(srcPath):
        shutil.copy(src=srcPath, dst=dstPath)
        #print("[UtilLib] - Success : copy file {} -> {}".format(srcPath, dstPath))
    else:
        #print("[UtilLib] - Fail : Source File is not exist.\n\t=>{}".format(srcPath))
        pass

def moveFile(srcPath, dstPath):
    """
    * 파일 이동
    :param srcPath: 원본 파일 
    :param dstPath: 대상 파일
    :return: 없음
    """
    if isExist(srcPath):
        shutil.move(src=srcPath, dst=dstPath)
        #print("[UtilLib] - Success : move file")
    else:
        #print("[UtilLib] - Fail : Source File is not exist.\n\t=>{}".format(srcPath))
        pass

def deleteFile(srcPath):
    """
    * 파일 삭제
    :param srcPath: 원본 파일 
    :return: 없음
    """
    if isExist(srcPath):
        os.remove(srcPath)
        print("[UtilLib] - Success : delete file")
    else:
        print("[UtilLib] - Fail : Source File is not exist.\n\t=>{}".format(srcPath))

def copyDir(srcPath, dstPath):
    """
    * 폴더 복사
    :param srcPath: 원본 폴더 
    :param dstPath: 대상 폴더
    :return: 
    """
    if isExist(srcPath):
        shutil.copytree(src=srcPath, dst=dstPath)
        print("[UtilLib] - Success : copy directory({})".format(srcPath))
    else:
        print("[UtilLib] - Fail : Source directory is not exist.\n\t=>{}".format(srcPath))

def moveDir(srcPath, dstPath):
    """
    * 폴더 이동
    :param srcPath: 원본 폴더 
    :param dstPath: 대상 폴더
    :return: 없음
    """
    if isExist(srcPath):
        shutil.move(src=srcPath, dst=dstPath)
        print("[UtilLib] - Success : move directory")
    else:
        print("[UtilLib] - Fail : Source directory is not exist.\n\t=>{}".format(srcPath))

def deleteDir(srcPath):
    """
    * 폴더 삭제
    :param srcPath: 원본 폴더 
    :return: 없음
    """
    if isExist(srcPath):
        shutil.rmtree(srcPath)
        print("[UtilLib] - Success : delete directory({})".format(srcPath))
    else:
        print("[UtilLib] - Fail : Source directory is not exist.\n\t=>{}".format(srcPath))

def createDir(srcPath):
    """
    * 폴서 생성
    :param srcPath: 원본 폴더 
    :return: 없음
    """
    if isExist(srcPath):
        #print("[UtilLib] - Fail : Directory is already exist.")
        return False
    else:
        os.mkdir(srcPath)
        #print("[UtilLib] - Success : create directory")
        return True

#------------------------------------------------------------------------------------------
# 자료형(list, dict) 값 확인
def printListData(listVar):
    for index, value in enumerate(listVar):
        try:
            print("\tIndex : {:<40} || value : {:<50}".format(index, value))
        except:
            print("\tIndex : {:<40} || value : ".format(index), value)


def printDictData(dictVar):
    for index, key in enumerate(dictVar):
        if type(dictVar[key]) is str:
            print("\tIndex : {:<16} || key : {:<16} || value :".format(index, key), dictVar[key])
        else:
            print("\tIndex : {:<16} || key : {:<16} || value : {}".format(index, key, dictVar[key]))

#------------------------------------------------------------------------------------------
# decimal 사칙연산
def decimal_add(value1 = None, value2 = None, dataList = None):
    """
    * 소수점 덧셈
    :param value1: 입력값 1
    :param value2: 입력값 2
    :param dataList: 입력데이터리스트 (float형 데이터를 포함한 리스트)
    :return: 연산 결과 (String 타입)
    """
    if value1 is None and value2 is None and dataList is None :
        return None
    else:
        if dataList is not None :
            result = 0.0
            for num in dataList:
                #print(type(num))
                result = Decimal(result) + Decimal(num)

            return str(result)
        else:
            result = Decimal(value1)+Decimal(value2)

            return str(result)

#print(decimal_add(dataList = [1,2,3,4,5]))

def decimal_subtract(value1 = None, value2 = None, dataList = None):
    """
    * 소수점 뺄셈
    :param value1: 입력값 1
    :param value2: 입력값 2
    :param dataList: 입력데이터리스트 (float 형 데이터를 포함한 리스트)
    :return: 연산 결과 (String 타입)
    """
    if value1 is None and value2 is None and dataList is None:
        return None
    else :
        if dataList is not None:
            result = 0.0
            for num in dataList:
                result = Decimal(result) - Decimal(num)
            return str(result)
        else:
            result = Decimal(value1) - Decimal(value2)
            return str(result)


def decimal_multiply(value1=None, value2=None, dataList = None):
    """
    * 소수점 곱셈
    :param value1: 입력값1
    :param value2: 입력값2
    :param dataList: 입력데이터리스트(float 형 데이터를 포함한 리스트)
    :return: 연산 결과 (String 타입)
    """
    if value1 is None and value2 is None and dataList is None:
        return None
    else:
        if dataList is not None:
            result = 1
            for num in dataList:
                result = Decimal(result) * Decimal(num)
            return str(result)
        else:
            result = Decimal(value1) * Decimal(value2)
            return str(result)

def decimal_divide(value1=None, value2=None, dataList = None):
    """
    * 소수점 나누기
    :param value1: 입력값 1
    :param value2: 입력값 2
    :param dataList: 입력데이터리스트 (float 형 데이터를 포함한 리스트)
    :return: 연산 결과 (String 타입)
    """
    if value1 is None and value2 is None and dataList is None:
        return None
    else:


        #print("length : {}".format(dataListSize))

        if dataList is not None:
            dataListSize = len(dataList)
            result = dataList[0]
            #print("first elem : {}".format(result))

            if 0 in dataList:
                return None

            for index in range(1, dataListSize):
                result = Decimal(result) / Decimal(dataList[index])

            return str(result)

        else:
            if value2 == 0:
                return None
            else:
                result = Decimal(value1) / Decimal(value2)
                return str(result)

#------------------------------------------------------------------------------------------
# 소수점
def getRound(value, digit):
    """
    * digit 자리수 아래로 버림
    :param value:입력 값
    :param digit:소수점 버릴 자리수
    :return:
    """
    tokens = splitToken(str(value), ".")
    newValue = tokens[0]+"."+tokens[1][:digit]
    return float(newValue)

def convert8digit(value):
    """
    * 소수점 8자리 문자열로 반환
    :param value:
    :return: 소수저머 8자리 문자열
    """
    fValue = float(value)

    return "{:.8f}".format(fValue)
#------------------------------------------------------------------------------------------
# 실수형 range 함수
def floatRange(start, end, step):
    """
    * 실수형 range 기능
    :param start: 시작 하는 실수 값
    :param end: 종료 하는 실수 값
    :param step: 증가 하는 실수 값
    :return: range()와 동일하게 쓰임
    """
    # ex) for hz in self.floatRange(start, end, step):
    r = start
    while(r<end):
        yield r
        r+=step
# ------------------------------------------------------------------------------------------
def getDuplicatedItemIndex(dataList):
    """
    * 리스트 데이터에 중복된 데이터의 인덱스를 반환
    :param dataList: 입력 데이터 리스트
    :return: finalResult(중복된 결과만 dict) | result (중복 전체 결과 dict)
    """
    result = defaultdict(list)

    for k, item in enumerate(dataList):
        result[item].append(k)

    finalResult = {key: value for key, value in result.items() if len(value) > 1}

    return finalResult, result











