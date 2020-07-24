from pypinyin import pinyin
import pypinyin
import pkg_resources
import pickle

class FourCornerMethod(object):
    def __init__(self, path):
        data_file = pkg_resources.resource_filename(__name__, path+"/data.pkl")
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
    
    def query(self, input_char, default=None):
        return self.data.get(input_char, default)

class ssc():
	def __init__(self, path):
		self.fcm = FourCornerMethod(path)
		self.ssc_encode_way = 'ALL'#'ALL','SOUND','SHAPE'

		self.yunmuDict = {'a':'1', 'o':'2', 'e':'3', 'i':'4', 
		             'u':'5', 'v':'6', 'ai':'7', 'ei':'7', 
		             'ui':'8', 'ao':'9', 'ou':'A', 'iou':'B',#有：you->yiou->iou->iu
		             'ie':'C', 've':'D', 'er':'E', 'an':'F', 
		             'en':'G', 'in':'H', 'un':'I', 'vn':'J',#晕：yun->yvn->vn->ven
		             'ang':'F', 'eng':'G', 'ing':'H', 'ong':'K'}

		self.shengmuDict = {'b':'1', 'p':'2', 'm':'3', 'f':'4', 
		             'd':'5', 't':'6', 'n':'7', 'l':'7', 
		             'g':'8', 'k':'9', 'h':'A', 'j':'B',
		             'q':'C', 'x':'D', 'zh':'E', 'ch':'F',
		             'sh':'G', 'r':'H', 'z':'E', 'c':'F', 
		             's':'G', 'y':'I', 'w':'J', '0':'0'}

		self.shapeDict = {'⿰':'1','⿱':'2','⿲':'3','⿳':'4','⿴':'5',#左右结构、上下、左中右、上中下、全包围
		                  '⿵':'6','⿶':'7','⿷':'8','⿸':'9','⿹':'A',#上三包、下三包、左三包、左上包、右上包
		                  '⿺':'B','⿻':'C', '0':'0'}#左下包、镶嵌、独体字：0

		self.strokesDict = {1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A',
		               11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K',
		               21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T', 30:'U',
		               31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z', 0:'0'}

		self.strokesDictReverse = {'1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'A':10,
		               'B':11, 'C':12, 'D':13, 'E':14, 'F':15, 'G':16, 'H':17, 'I':18, 'J':19, 'K':20,
		               'L':21, 'M':22, 'N':23, 'O':24, 'P':25, 'Q':26, 'R':27, 'S':28, 'T':29, 'U':30,
		               'V':31, 'W':32, 'X':33, 'Y':34, 'Z':35, '0':0}

		self.soundWeight=0.5
		self.shapeWeight=0.5
		self.hanziStrokesDict = self.getHanziStrokesDict(path)
		self.hanziStructureDict = self.getHanziStructureDict(path)
		self.hanziSSCDict = self.getHanziSSCDict(path)

	def getHanziStrokesDict(self,path):
	    hanziStrokesDict = {}#汉子：笔画数
	    strokes_filepath = pkg_resources.resource_filename(__name__, path+"/utf8_strokes.txt")
	    with open(strokes_filepath, 'r', encoding='UTF-8') as f:#文件特征：
	        for line in f:
	            line = line.split()
	            hanziStrokesDict[line[1]]=line[2]
	    return hanziStrokesDict

	def getHanziStructureDict(self,path):
	    hanziStructureDict = {}#汉子：形体结构
	    structure_filepath = pkg_resources.resource_filename(__name__, path+"/unihan_structure.txt")
	    with open(structure_filepath, 'r', encoding='UTF-8') as f:#文件特征：U+4EFF\t仿\t⿰亻方\n
	        for line in f:
	            line = line.split()
	            if line[2][0] in self.shapeDict:
	                hanziStructureDict[line[1]]=line[2][0]
	    return hanziStructureDict

	def getHanziSSCDict(self,path):
	    hanziSSCDict = {}#汉子：SSC码   
	    hanzi_ssc_filepath = pkg_resources.resource_filename(__name__, path+"/hanzi_ssc_res.txt")
	    with open(hanzi_ssc_filepath, 'r', encoding='UTF-8') as f:#文件特征：U+4EFF\t仿\t音形码\n
	        for line in f:
	            line = line.split()
	            hanziSSCDict[line[1]]=line[2]  
	    return hanziSSCDict 

	def getSoundCode(self, one_chi_word):
	    res = []
	    shengmuStr = pinyin(one_chi_word, style=pypinyin.INITIALS, heteronym=False, strict=False)[0][0]
	    if shengmuStr not in self.shengmuDict:
	        shengmuStr = '0'
	    
	    yunmuStrFullStrict = pinyin(one_chi_word, style=pypinyin.FINALS_TONE3, heteronym=False, strict=True)[0][0]

	    yindiao = '0'
	    if yunmuStrFullStrict[-1] in ['1','2','3','4']:
	        yindiao = yunmuStrFullStrict[-1]
	        yunmuStrFullStrict = yunmuStrFullStrict[:-1]

	    if yunmuStrFullStrict in self.yunmuDict:
	        #声母，韵母辅音补码，韵母，音调
	        res.append(self.yunmuDict[yunmuStrFullStrict])
	        res.append(self.shengmuDict[shengmuStr])
	        res.append('0')
	    elif len(yunmuStrFullStrict)>1:
	        res.append(self.yunmuDict[yunmuStrFullStrict[1:]])
	        res.append(self.shengmuDict[shengmuStr])
	        res.append(self.yunmuDict[yunmuStrFullStrict[0]])
	    else:
	        res.append('0')
	        res.append(self.shengmuDict[shengmuStr])
	        res.append('0')
	        
	    res.append(yindiao)
	    return res

	def getShapeCode(self, one_chi_word):
	    res = []
	    structureShape = self.hanziStructureDict.get(one_chi_word, '0')#形体结构
	    res.append(self.shapeDict[structureShape])
	    
	    fourCornerCode = self.fcm.query(one_chi_word)#四角号码（5位数字）
	    if fourCornerCode is None:
	        res.extend(['0', '0', '0', '0', '0'])
	    else:
	        res.extend(fourCornerCode[:])
	    
	    strokes = self.hanziStrokesDict.get(one_chi_word, '0')#笔画数
	    if int(strokes) >35:
	        res.append('Z')
	    else:
	        res.append(self.strokesDict[int(strokes)])     
	    return res       


	def getSSC(self, hanzi_sentence):
	    hanzi_sentence_ssc_list = []
	    for one_chi_word in hanzi_sentence:
	        ssc = self.hanziSSCDict.get(one_chi_word, None)
	        if ssc is None:
	            soundCode = self.getSoundCode(one_chi_word)
	            shapeCode = self.getShapeCode(one_chi_word)
	            ssc = "".join(soundCode+shapeCode)
	        if self.ssc_encode_way=="SOUND":
	            ssc=ssc[:4]
	        elif self.ssc_encode_way=="SHAPE":
	            ssc=ssc[4:]
	        else:
	            pass
	        hanzi_sentence_ssc_list.append(ssc)
	    return hanzi_sentence_ssc_list

	def computeSoundCodeSimilarity(self, soundCode1, soundCode2):#soundCode=['2', '8', '5', '2']
	    featureSize=len(soundCode1)
	    wights=[0.4,0.4,0.1,0.1]
	    multiplier=[]
	    for i in range(featureSize):
	        if soundCode1[i]==soundCode2[i]:
	            multiplier.append(1)
	        else:
	            multiplier.append(0)
	    soundSimilarity=0
	    for i in range(featureSize):
	        soundSimilarity += wights[i]*multiplier[i]
	    return soundSimilarity
	    
	def computeShapeCodeSimilarity(self, shapeCode1, shapeCode2):#shapeCode=['5', '6', '0', '1', '0', '3', '8']
	    featureSize=len(shapeCode1)
	    wights=[0.25,0.1,0.1,0.1,0.1,0.1,0.25]
	    multiplier=[]
	    for i in range(featureSize-1):
	        if shapeCode1[i]==shapeCode2[i]:
	            multiplier.append(1)
	        else:
	            multiplier.append(0)
	    multiplier.append(1- abs(self.strokesDictReverse[shapeCode1[-1]]-self.strokesDictReverse[shapeCode2[-1]])*1.0 / max(self.strokesDictReverse[shapeCode1[-1]],self.strokesDictReverse[shapeCode2[-1]]) )
	    shapeSimilarity=0
	    for i in range(featureSize):
	        shapeSimilarity += wights[i]*multiplier[i]
	    return shapeSimilarity

	def computeSSCSimilarity(self,ssc1, ssc2):
	    #return 0.5*computeSoundCodeSimilarity(ssc1[:4], ssc2[:4])+0.5*computeShapeCodeSimilarity(ssc1[4:], ssc2[4:])

	    if self.ssc_encode_way=="SOUND":
	        return self.computeSoundCodeSimilarity(ssc1, ssc2)
	    elif self.ssc_encode_way=="SHAPE":
	        return self.computeShapeCodeSimilarity(ssc1, ssc2)
	    else:
	        soundSimi=self.computeSoundCodeSimilarity(ssc1[:4], ssc2[:4])
	        shapeSimi=self.computeShapeCodeSimilarity(ssc1[4:], ssc2[4:])
	        return self.soundWeight*soundSimi+self.shapeWeight*shapeSimi

	def compute_similarity(self, ssc1, ssc2):
	    score = 1
	    ssc1 = self.getSSC(ssc1)
	    ssc2 = self.getSSC(ssc2)
	    
	    for idx in range(max(len(ssc1), len(ssc2))):
	        if idx<min(len(ssc1), len(ssc2)):
	            tmp = self.computeSSCSimilarity(ssc1[idx], ssc2[idx])
	            if tmp!=1:
	                score*=tmp
	    return score