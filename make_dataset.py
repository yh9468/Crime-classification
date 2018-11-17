import os
import shutil

arr = []
arrP = []

class person:
    def __init__(self, id, how):
        self.id = id
        self.how = how

    def getid(self):
        return self.id

    def gethow(self):
        return self.how

def search(dirname):
    count = 0
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        flag = False
        for temp in arrP:
            if(temp.getid() == filename[:7]):
                flag = True
                if(os.path.isdir('dataset/test/'+temp.gethow())):
                    shutil.copy('dataset/test/'+filename, 'dataset/test/'+temp.gethow()+'/'+filename)
                else:
                    os.mkdir('dataset/test/'+temp.gethow())
                    shutil.copy('dataset/test/'+filename, 'dataset/test/'+temp.gethow()+'/'+filename)

                break
        if(flag == False):
            if os.path.isfile(full_filename):
                print (filename)
                os.remove(full_filename)
                count = count+1
    print("total count : {}".format(count))

def readtxt(dirname):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        f = open(full_filename, 'r')
        lines = f.readlines()
        for temp in lines:
            if(len(temp) < 15):
                continue
            temp = temp.split(',')
            if(temp[2] == "Probation Violation" or temp[2] == "Liquor Violation"
                    or temp[2] == "Courts and Civil Proceedings Violations"
                    or temp[2] == "Other" or temp[2] == "Profession and Occupation Violations"):
                continue
            newone = person(temp[0],temp[2])
            arrP.append(newone)

def writetxt():
    f = open("new_dataset.txt", 'w')
    for element in arrP:
        newstring = element.getid()+','+element.gethow()+'\n'
        f.write(newstring)
    f.close()

def changedir(str):
    filepath = "dataset/"+str
    filelist = os.listdir(filepath)
    if os.path.isdir("dataset/"+str+ "low"):
        return
    else:
        os.mkdir("dataset/"+str+"/low")
        os.mkdir("dataset/"+str+"/middle")
        os.mkdir("dataset/"+str+"/high")

    for dirname in filelist:
        temp_filelist = os.listdir(filepath +"/"+dirname)
        if(dirname =="Eavesdropping and Communication" or dirname == "Offenses against Public Order"
        or dirname == "Transportation Violations" or dirname == "Criminal Damage"
        or dirname == "Failure to Appear" or dirname == "DUI" or dirname == "Weapons and Explosives"
        or dirname == "County Regulations Violations" or dirname == "Family Offenses"
        or dirname == "ANIMAL CRUELTY" or dirname == "Interfere with Judicial Process"):
            for filename in temp_filelist:
                shutil.copy(filepath +"/"+dirname + '/' + filename, filepath+'/' + "low" + '/' + filename)

        elif(dirname == "Forgery" or dirname =="Fraud" or dirname == "Robbery" or
        dirname == "Sex Crimes" or dirname == "Obstruction" or dirname == "Drug Offenses"
        or dirname == "Theft"):
            for filename in temp_filelist:
                shutil.copy(filepath +"/"+dirname + '/' + filename, filepath+'/' + "middle" + '/' + filename)

        else:
            for filename in temp_filelist:
                shutil.copy(filepath +"/"+dirname + '/' + filename, filepath+'/' + "high" + '/' + filename)



changedir("test")
#readtxt("dataset/txt")
#search("dataset/test")
writetxt()