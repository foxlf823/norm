# -*- coding: UTF-8 -*-
import os
import argparse
import codecs
import re
import shutil
from jpype import *

class Util:
    @classmethod
    def setMap(self, keyValueListMap, key, value):
        valueList = keyValueListMap.get(key)
        if valueList == None:
            valueList = list()
            keyValueListMap[key] = valueList
        valueList = Util.setList(valueList, value);
        return keyValueListMap

    @classmethod
    def setList(self, listt, value):
        if (value not in listt) and (value != u""):
            listt.append(value)
        return listt

    @classmethod
    def firstIndexOf(self, tokens, i, pattern):
        while i >=0:
            if re.match(pattern+ur".*", tokens[i]):
                i -= 1
                return i
            i -= 1
        return -1

    @classmethod
    def read(self, file_path):
        with codecs.open(file_path, 'r', 'UTF-8') as fp:
            return fp.read()

    @classmethod
    def containsAny(self, first, second):
        first_ = set(first)
        second_ = set(second)

        return len(first_ & second_) != 0


class Abbreviation:
    wikiAbbreviationExpansionListMap = dict()


    def __init__(self):
        self.textAbbreviationExpansionMap = dict()

    @classmethod
    def setWikiAbbreviationExpansionMap(self, file_path):
        with codecs.open(file_path, 'r', 'UTF-8') as fp:
            for line in fp:
                line = line.strip()
                token = re.split(ur"\|\|", line)
                Abbreviation.wikiAbbreviationExpansionListMap = Util.setMap(Abbreviation.wikiAbbreviationExpansionListMap, token[0].lower(), token[1].lower())
    @classmethod
    def getTentativeExpansion(self, tokens, i, abbreviationLength):
        expansion = u""
        while (i >= 0 and abbreviationLength > 0):
            expansion = tokens[i]+" "+expansion
            i -= 1
            abbreviationLength -= 1

        return expansion.strip()

    @classmethod
    def getExpansionByHearstAlgorithm(self, shortForm, longForm):
        sIndex = len(shortForm) - 1
        lIndex = len(longForm) - 1

        while(sIndex >= 0):
            currChar = shortForm[sIndex].lower()
            if not currChar.isalnum():
                sIndex -= 1
                continue

            while (((lIndex >= 0) and
                    (longForm[lIndex].lower() != currChar)) or
                    ((sIndex == 0) and (lIndex > 0) and
                    (longForm[lIndex-1].isalnum()))):
                lIndex -= 1

            if lIndex < 0:
                return u""

            lIndex -= 1
            sIndex -= 1

        lIndex = longForm.rfind(u" ", lIndex) + 1
        longForm = longForm[lIndex:]

        return longForm

    @classmethod
    def getEntireAbbreviation(self, text, string, indexes):
        if len(indexes) != 2:
            return string
        begin = int(indexes[0])
        end = int(indexes[1])
        if re.match(ur"(^|\s|\W)[a-zA-Z]/"+string+ur"/[a-zA-Z](\s|$|\W)", text[begin-3, end+3].lower()) :
            return text[begin-2, end+2].lower()
        elif re.matches(ur"(^|\s|\W)"+string+ur"/[a-zA-Z]/[a-zA-Z](\s|$|\W)", text[begin-1, end+5].lower()):
            return text[begin, end+4].lower()
        elif re.matches(ur"(^|\s|\W)[a-zA-Z]/[a-zA-Z]/"+string+ur"(\s|$|\W)", text[begin-5, end+1].lower()):
            return text[begin-4, end].lower()
        return string

    @classmethod
    def getBestExpansion(self, text, expansionList):
        maxNumberOfContentWords = 0
        maxContainedContentWords = 0
        returnExpansion = u""
        for expansion in expansionList:
            expansionContentWordsList = Ling.getContentWordsList(re.split(ur"\s", expansion))
            tempNumberOfContentWords = len(expansionContentWordsList)
            tempContainedContentWords = 0
            for  expansionContentWord in expansionContentWordsList:
                if text.find(u" " + expansionContentWord) != -1 or text.find(expansionContentWord + u" ") != -1:
                    tempContainedContentWords += 1

            if tempNumberOfContentWords > maxNumberOfContentWords and tempContainedContentWords == tempNumberOfContentWords:
                maxNumberOfContentWords = tempNumberOfContentWords
                maxContainedContentWords = 1000
                returnExpansion = expansion
            elif tempNumberOfContentWords >= maxNumberOfContentWords and tempContainedContentWords > maxContainedContentWords:
                maxNumberOfContentWords = tempNumberOfContentWords
                maxContainedContentWords = tempContainedContentWords
                returnExpansion = expansion

        return returnExpansion

    @classmethod
    def getTrimmedExpansion(self, text, string, indexes, expansion):
        if len(indexes) != 2:
            return string
        begin = int(indexes[0])
        end = int(indexes[1])
        if re.matches(ur"(^|\s|\W)[a-zA-Z]/"+string+ur"/[a-zA-Z](\s|$|\W)", text[begin-3, end+3].lower()):
            return expansion[1].lower()
        elif re.matches(ur"(^|\s|\W)"+string+ur"/[a-zA-Z]/[a-zA-Z](\s|$|\W)", text[begin-1, end+5].lower()):
            return expansion[0].lower()
        elif re.matches(ur"(^|\s|\W)[a-zA-Z]/[a-zA-Z]/"+string+ur"(\s|$|\W)", text[begin-5, end+1].lower()):
            return expansion[2].lower()
        return string

    @classmethod
    def getAbbreviationExpansion(self, abbreviationObject, text, string, indexes):
        shortForm_longForm_map = abbreviationObject.getTextAbbreviationExpansionMap()
        stringTokens = re.split(ur"\s", string)

        if len(stringTokens) == 1 and len(stringTokens[0]) == 1 :
            stringTokens[0] = Abbreviation.getEntireAbbreviation(text, string, re.split(ur"\|", indexes))
        newString = u""

        for stringToken in stringTokens:
            if stringToken in shortForm_longForm_map:
                newString += shortForm_longForm_map.get(stringToken)+u" "
                continue
            candidateExpansionsList = Abbreviation.wikiAbbreviationExpansionListMap.get(stringToken) if stringToken in Abbreviation.wikiAbbreviationExpansionListMap else None

            if candidateExpansionsList == None:
                newString += stringToken + u" "
            else :
                expansion = candidateExpansionsList[0] if len(candidateExpansionsList) == 1 else Abbreviation.getBestExpansion(text, candidateExpansionsList)
                if expansion == u"":
                    newString += stringToken + u" "
                else:
                    newString += expansion + u" "

        if len(stringTokens) == 1 and stringTokens[0] != string:
            newString = getTrimmedExpansion(text, string, re.split(ur"\|", indexes), re.split(ur"/", newString))

        newString = newString.strip()
        return u"" if newString == (string) else newString


    def setTextAbbreviationExpansionMap_(self, tokens, abbreviationLength, abbreviation, expansionIndex):
        expansion = Abbreviation.getTentativeExpansion(tokens, expansionIndex, abbreviationLength)
        expansion = Abbreviation.getExpansionByHearstAlgorithm(abbreviation, expansion).lower().strip()
        if expansion != u"":
            self.textAbbreviationExpansionMap[abbreviation] = expansion


    def setTextAbbreviationExpansionMap (self, file_path):
        with codecs.open(file_path, 'r', 'UTF-8') as fp:
            for line in fp:
                line = line.strip()
                tokens = re.split(ur"\s+", line)
                size = len(tokens)
                for i in range(size):
                    expansionIndex = -1

                    if (re.match(ur"\(\w+(\-\w+)?\)(,|\.)?", tokens[i])) or (re.match(ur"\([A-Z]+(;|,|\.)", tokens[i])):
                        expansionIndex = i - 1
                    elif re.match(ur"[A-Z]+\)", tokens[i]):
                        expansionIndex = Util.firstIndexOf(tokens, i, ur"\(")

                    if expansionIndex == -1:
                        continue

                    abbreviation = tokens[i].replace(u"(", u"").replace(u")", u"").lower()
                    reversedAbbreviation = Ling.reverse(abbreviation)

                    if abbreviation[len(abbreviation) - 1] == u',' or abbreviation[len(abbreviation) - 1] == u'.' or abbreviation[len(abbreviation) - 1] == u';':
                        abbreviation = abbreviation[0: len(abbreviation) - 1]

                    if (abbreviation in self.textAbbreviationExpansionMap) or (reversedAbbreviation in self.textAbbreviationExpansionMap):
                        continue

                    abbreviationLength = len(abbreviation)
                    self.setTextAbbreviationExpansionMap_(tokens, abbreviationLength, abbreviation, expansionIndex)
                    if abbreviation not in self.textAbbreviationExpansionMap:
                        self.setTextAbbreviationExpansionMap_(tokens, abbreviationLength, reversedAbbreviation, expansionIndex)

    def getTextAbbreviationExpansionMap(self):
        return self.textAbbreviationExpansionMap

class Ling:
    stopwords = set()
    digitToWordMap = dict()
    wordToDigitMap = dict()
    suffixMap = dict()
    prefixMap = dict()
    affixMap = dict()
    startJVM(getDefaultJVMPath(), "-ea", "-Dfile.encoding=UTF-8", "-Djava.class.path={}".format(os.path.abspath(".")))
    PorterStemmer = JClass("PorterStemmer")

    def __init__(self):
        pass

    @classmethod
    def setStopwordsList(self, file_path):
        with codecs.open(file_path, 'r', 'UTF-8') as fp:
            for line in fp:
                line = line.strip()
                if line == u'':
                    continue
                Ling.stopwords.add(line)

    @classmethod
    def getStopwordsList(self):
        return Ling.stopwords

    @classmethod
    def setDigitToWordformMapAndReverse(self, file_path):
        with codecs.open(file_path, 'r', 'UTF-8') as fp:
            for line in fp:
                line = line.strip()
                tokens = re.split(ur"\|\|", line)
                Ling.digitToWordMap = Util.setMap(Ling.digitToWordMap, tokens[0], tokens[1]);
                Ling.wordToDigitMap[tokens[1]]=tokens[0]

    @classmethod
    def setSuffixMap(self, file_path):
        with codecs.open(file_path, 'r', 'UTF-8') as fp:
            for line in fp:
                line = line.strip()
                tokens = re.split(ur"\|\|", line)
                if len(tokens) == 1:
                    values = Ling.suffixMap.get(tokens[0])
                    if values == None:
                        values = list()
                        Ling.suffixMap[tokens[0]]=values
                else:
                    Ling.suffixMap = Util.setMap(Ling.suffixMap, tokens[0], tokens[1])

    @classmethod
    def setPrefixMap(self, file_path):
        with codecs.open(file_path, 'r', 'UTF-8') as fp:
            for line in fp:
                line = line.strip()
                tokens = re.split(ur"\|\|", line)
                value = u"" if len(tokens) == 1 else tokens[1]
                Ling.prefixMap[tokens[0]] = value

    @classmethod
    def setAffixMap(self, file_path):
        with codecs.open(file_path, 'r', 'UTF-8') as fp:
            for line in fp:
                line = line.strip()
                tokens = re.split(ur"\|\|", line)
                value = u"" if len(tokens) == 1 else tokens[1]
                Ling.affixMap[tokens[0]] = value

    @classmethod
    def getStemmedPhrase(self, string):

        stemmed_name = u""
        str_tokens = re.split(ur"\s+", string)
        for token in str_tokens:
            if token in Ling.stopwords:
                stemmed_name += token + u" "
                continue

            stemmed_token = Ling.PorterStemmer.get_stem(token).strip()
            if stemmed_token == u"":
                stemmed_token = token
            stemmed_name += stemmed_token + u" "

        stemmed_name = stemmed_name.strip()


        return stemmed_name

    @classmethod
    def reverse(self, string):
        reversedString = u""
        size = len(string)-1
        for i in range(size, -1, -1):
            reversedString += string[i]

        return reversedString

    @classmethod
    def getContentWordsList(self, words):
        contentWordsList = list()
        for word in words:
            if word in Ling.stopwords:
                continue
            contentWordsList = Util.setList(contentWordsList, word)

        return contentWordsList



class Terminology:

    def __init__(self):
        self.cuiAlternateCuiMap = dict()
        self.nameToCuiListMap = dict()
        self.cuiToNameListMap = dict()
        self.stemmedNameToCuiListMap = dict()
        self.cuiToStemmedNameListMap = dict()
        self.tokenToNameListMap = dict()
        self.compoundNameToCuiListMap = dict()
        self.simpleNameToCuiListMap = dict()

    def getNameToCuiListMap(self):

        return self.nameToCuiListMap


    def getCuiAlternateCuiMap(self):
        return self.cuiAlternateCuiMap

    # 如果concept id首字母是字母，则为preferredID
    # 否则设置第一个为preferredID
    # preferred id和altID可能重复
    def get_preferredID_set_altID(self, identifiers):
        set = False
        altIDs = list()

        for i, _ in enumerate(identifiers):
            if i == 0:
                preferredID = identifiers[i]
            if identifiers[i][0].isalpha() and set == False:
                preferredID = identifiers[i]
                set = True
                continue

            altIDs.append(identifiers[i])

        if len(altIDs) != 0:
            self.cuiAlternateCuiMap[preferredID] = altIDs

        return preferredID

    def loadMaps(self, conceptName, cui):
        self.nameToCuiListMap = Util.setMap(self.nameToCuiListMap, conceptName, cui)
        self.cuiToNameListMap = Util.setMap(self.cuiToNameListMap, cui, conceptName)

        stemmedConceptName = Ling.getStemmedPhrase(conceptName)
        self.stemmedNameToCuiListMap = Util.setMap(self.stemmedNameToCuiListMap, stemmedConceptName, cui)
        self.cuiToStemmedNameListMap = Util.setMap(self.cuiToStemmedNameListMap, cui, stemmedConceptName)

        conceptNameTokens = re.split(ur"\s+", conceptName)
        for conceptNameToken in conceptNameTokens:
            if conceptNameToken in Ling.getStopwordsList():
                continue

            self.tokenToNameListMap = Util.setMap(self.tokenToNameListMap, conceptNameToken, conceptName);

        if cui.find(u"|") != -1: # 当使用训练数据作为termininology是才进入，对应composite mention
            self.nameToCuiListMap.pop(conceptName)
            self.stemmedNameToCuiListMap.pop(stemmedConceptName)
            for conceptNameToken in conceptNameTokens:
                if conceptNameToken in Ling.getStopwordsList():
                    continue
                name_list = self.tokenToNameListMap.get(conceptNameToken)
                new_name_list = list()
                for name in name_list:
                    if name == conceptName:
                        continue
                    new_name_list.append(name)
                self.tokenToNameListMap[conceptNameToken] = new_name_list

            # composite mention存入
            self.compoundNameToCuiListMap = Util.setMap(self.compoundNameToCuiListMap, conceptName, cui)


    def loadTerminology(self, path):

        with codecs.open(path, 'r', 'UTF-8') as fp:
            for line in fp:
                line = line.strip()
                if line == u'':
                    continue
                token = re.split(ur"\|\|", line)
                cui = self.get_preferredID_set_altID(re.split(ur"\|",token[0])) if token[0].find(u"|") != -1 else token[0]

                conceptNames = re.split(ur"\|", token[1].lower())

                for conceptName in conceptNames:
                    self.loadMaps(conceptName, cui)

    def loadTrainingDataTerminology(self, path):

        for input_file_name in os.listdir(path):
            if input_file_name.find(".concept") == -1:
                continue
            input_file_path = os.path.join(path, input_file_name)

            with codecs.open(input_file_path, 'r', 'UTF-8') as fp:
                for line in fp:
                    line = line.strip()
                    tokens = re.split(ur"\|\|", line)
                    conceptName = tokens[3].lower().strip()
                    cuis = re.split(ur"\+", tokens[4]) if tokens[4].find("+") != -1 else re.split(ur"\|", tokens[4])
                    MeSHorSNOMEDcuis = Terminology.getMeSHorSNOMEDCuis(cuis)
                    if MeSHorSNOMEDcuis != u"":
                        self.loadMaps(conceptName, MeSHorSNOMEDcuis)
                    self.setOMIM(tokens[4], MeSHorSNOMEDcuis, conceptName)

                    cui = MeSHorSNOMEDcuis if MeSHorSNOMEDcuis != u"" else tokens[4].replace(u"OMIM:", u"")

                    simpleConceptNames = SimpleNameSieve.getTerminologySimpleNames(re.split(ur"\s+", conceptName))
                    for simpleConceptName in simpleConceptNames:
                        self.simpleNameToCuiListMap = Util.setMap(self.simpleNameToCuiListMap, simpleConceptName, cui)



    # for NCBI data, 得到Mesh ID，去除mention的OMIM ID
    @classmethod
    def getMeSHorSNOMEDCuis(self, cuis):
        cuiStr = u""
        for cui in cuis:
            if cui.find(u"OMIM") != -1:
                continue
            cuiStr = cui if cuiStr == u"" else cuiStr+u"|"+cui

        return cuiStr

    @classmethod
    def getOMIMCuis(self, cuis):
        OMIMcuis = list()
        for cui in cuis:
            if cui.find(u"OMIM") == -1:
                continue
            cui = re.split(u":", cui)[1]
            OMIMcuis = Util.setList(OMIMcuis, cui)
        return OMIMcuis

    def setOMIM(self, cuis, MeSHorSNOMEDcuis, conceptName):
        if MeSHorSNOMEDcuis == u"": # 如果Mesh ID为空，则用OMIM
            cuis = cuis.replace(u"OMIM:", u"")
            self.loadMaps(conceptName, cuis)

        else : # 否则用OMIM为候选ID
            cuis_arr = re.split(ur"\|", cuis)
            for cui in cuis_arr:
                if cui.find(u"OMIM") == -1:
                    continue
                cui = re.split(u":", cui)[1]
                self.cuiAlternateCuiMap = Util.setMap(self.cuiAlternateCuiMap, MeSHorSNOMEDcuis, cui)

class Concept:
    def __init__(self, indexes, name, goldMeSHorSNOMEDCui, goldOMIMCuis):
        self.indexes = indexes
        self.name = name.lower().strip()
        self.goldMeSHorSNOMEDCui = goldMeSHorSNOMEDCui
        self.goldOMIMCuis = goldOMIMCuis
        self.nameExpansion = None
        self.stemmedName = None
        self.cui = None
        self.alternateCuis = None
        self.normalizingSieveLevel = None

    def setNameExpansion(self, text, abbreviationObject):
        self.nameExpansion = Abbreviation.getAbbreviationExpansion(abbreviationObject, text, self.name, self.indexes)

    def setStemmedName(self):
        self.stemmedName = Ling.getStemmedPhrase(self.name)

    def setCui(self, cui):
        self.cui = cui

    def getCui(self):
        return self.cui

    def setAlternateCuis(self, alternateCuis):

        self.alternateCuis = list()
        for alternateCui in alternateCuis:
            self.alternateCuis = Util.setList(self.alternateCuis, alternateCui)

    def setNormalizingSieveLevel(self, sieveLevel):
        self.normalizingSieveLevel = sieveLevel

    def getName(self):

        return self.name

    def getNormalizingSieve(self):
        return self.normalizingSieveLevel

    def getGoldMeSHorSNOMEDCui(self):
        return self.goldMeSHorSNOMEDCui

    def getGoldOMIMCuis(self):
        return self.goldOMIMCuis

    def getAlternateCuis(self):
        return self.alternateCuis






class DocumentConcepts:
    def __init__(self, filename, text):
        self.filename = filename
        self.text = text
        self.concepts = list()

    def setConcept(self, tokens, abbreviationObject):

        cuis = re.split(ur"\+", tokens[4]) if tokens[4].find("+") != -1 else re.split(ur"\|", tokens[4])
        MeSHorSNOMEDcuis = Terminology.getMeSHorSNOMEDCuis(cuis)
        OMIMcuis = Terminology.getOMIMCuis(cuis)
        concept = Concept(tokens[1], tokens[3], MeSHorSNOMEDcuis, OMIMcuis)
        concept.setNameExpansion(self.text, abbreviationObject)
        concept.setStemmedName()
        self.concepts.append(concept)

    def getConcepts(self):
        return self.concepts



class Documents:
    @classmethod
    def getDataSet(self, path):

        dataset = list()
        for input_file_name in os.listdir(path):
            if input_file_name.find(".concept") == -1:
                continue
            input_file_path = os.path.join(path, input_file_name)
            textFile = input_file_path.replace(".concept", ".txt")
            abbreviationObject = Abbreviation()
            abbreviationObject.setTextAbbreviationExpansionMap(textFile)
            documentConceptsObject = DocumentConcepts(textFile, Util.read(textFile))
            with codecs.open(input_file_path, 'r', 'UTF-8') as fp:
                for line in fp:
                    line = line.strip()
                    tokens = re.split(ur"\|\|", line)
                    documentConceptsObject.setConcept(tokens, abbreviationObject)
            dataset.append(documentConceptsObject)

        return dataset


class Evaluation:
    totalNames = 0
    tp = 0
    fp = 0
    accuracy = 0.0

    @classmethod
    def incrementTotal(self):
        Evaluation.totalNames += 1

    @classmethod
    def incrementTP(self):
        Evaluation.tp += 1

    @classmethod
    def incrementFP(self):
        Evaluation.fp += 1


    @classmethod
    def evaluateClassification(self, concept, concepts):
        Evaluation.incrementTotal()
        if (concept.getGoldMeSHorSNOMEDCui() != u"" and concept.getGoldMeSHorSNOMEDCui() == concept.getCui()) \
            or (len(concept.getGoldOMIMCuis()) != 0 and concept.getCui() in concept.getGoldOMIMCuis()):
            Evaluation.incrementTP()
        elif concept.getGoldMeSHorSNOMEDCui().find(u"|") != -1 and concept.getCui().find(u"|") != -1:
            gold = set(re.split(ur"\|", concept.getGoldMeSHorSNOMEDCui()))
            predicted = set(re.split(ur"\|", concept.getCui()))

            bFindPredictNotInGold = False
            for p in predicted:
                if p not in gold:
                    bFindPredictNotInGold = True
                    break
            if bFindPredictNotInGold:
                Evaluation.incrementFP()
            else:
                Evaluation.incrementTP()

            minus_set = gold - predicted
            if len(minus_set) == 0:
                Evaluation.incrementTP()
            else :
                Evaluation.incrementFP()

        elif concept.getAlternateCuis() is not None and len(concept.getAlternateCuis()) != 0 :
            if concept.getGoldMeSHorSNOMEDCui() != u"" and concept.getGoldMeSHorSNOMEDCui() in concept.getAlternateCuis() :
                Evaluation.incrementTP()
                concept.setCui(concept.getGoldMeSHorSNOMEDCui())

            elif len(concept.getGoldOMIMCuis()) != 0 and Util.containsAny(concept.getAlternateCuis(), concept.getGoldOMIMCuis()) :
                Evaluation.incrementTP();
                if len(concept.getGoldOMIMCuis()) == 1:
                    concept.setCui(concept.getGoldOMIMCuis()[0])

            else :
                Evaluation.incrementFP()
        else :
            Evaluation.incrementFP()

    @classmethod
    def computeAccuracy(self):

        Evaluation.accuracy = Evaluation.tp * 1.0 / Evaluation.totalNames

    @classmethod
    def printResults(self):

        print("*********************")
        print("Total Names: {}".format(Evaluation.totalNames))
        print("True Normalizations: {}".format(Evaluation.tp))
        print("False Normalizations: {}".format(Evaluation.fp))
        print("Accuracy: {}".format(Evaluation.accuracy))
        print("*********************")






class AmbiguityResolution:
    @classmethod
    def start(self, concepts, cuiNamesMap):
        for concept in concepts.getConcepts():
            if concept.getNormalizingSieve() != 1 or concept.getCui() == u"CUI-less":
                Evaluation.evaluateClassification(concept, concepts)
                continue

            conceptName = concept.getName()
            conceptNameTokens = re.split(ur"\s+", conceptName)

            trainingDataCuis = Sieve.getTrainingDataTerminology().getNameToCuiListMap().get(conceptName) \
                if conceptName in Sieve.getTrainingDataTerminology().getNameToCuiListMap() else None

            if trainingDataCuis is None or len(trainingDataCuis) == 1:
                Evaluation.evaluateClassification(concept, concepts)
                continue

            if len(conceptNameTokens) > 1:
                concept.setCui(u"CUI-less")
            else:
                countCUIMatch = 0
                for cui in trainingDataCuis:
                    names = cuiNamesMap.get(cui) if cui in cuiNamesMap else list()
                    for name in names:
                        nameTokens = re.split(ur"\s+", name)
                        if len(nameTokens) == 1:
                            continue
                        if re.match(conceptName+ur" .*", name):
                            countCUIMatch += 1



                if countCUIMatch > 0:
                    concept.setCui(u"CUI-less")

            Evaluation.evaluateClassification(concept, concepts)


class MultiPassSieveNormalizer:
    maxSieveLevel = 0

    def __init__(self):
        pass

    @classmethod
    def pass_(self, concept, currentSieveLevel):
        if concept.getCui() != u"":
            concept.setAlternateCuis(Sieve.getAlternateCuis(concept.getCui()))
            concept.setNormalizingSieveLevel(currentSieveLevel-1)

            return False


        if currentSieveLevel > MultiPassSieveNormalizer.maxSieveLevel:
            return False

        return True

    @classmethod
    def applyMultiPassSieve(self, concept):
        currentSieveLevel = 1

        # Sieve 1
        concept.setCui(Sieve.exactMatchSieve(concept.getName()))
        currentSieveLevel += 1
        if not MultiPassSieveNormalizer.pass_(concept, currentSieveLevel):
            return


class Sieve:
    standardTerminology = Terminology()
    trainingDataTerminology = Terminology()

    @classmethod
    def setStandardTerminology(self, dict_path):
        Sieve.standardTerminology.loadTerminology(dict_path)

    @classmethod
    def setTrainingDataTerminology(self, train_path):
        Sieve.trainingDataTerminology.loadTrainingDataTerminology(train_path)

    @classmethod
    def getAlternateCuis(self, cui):
        alternateCuis = list()
        if cui in Sieve.trainingDataTerminology.getCuiAlternateCuiMap():
            alternateCuis.extend(Sieve.trainingDataTerminology.getCuiAlternateCuiMap().get(cui))

        if cui in Sieve.standardTerminology.getCuiAlternateCuiMap():
            alternateCuis.extend(Sieve.standardTerminology.getCuiAlternateCuiMap().get(cui))

        return alternateCuis

    @classmethod
    def getTerminologyNameCui(self, nameToCuiListMap, name):
        return nameToCuiListMap.get(name)[0] if name in nameToCuiListMap and len(nameToCuiListMap.get(name)) == 1 else u""


    @classmethod
    def exactMatchSieve(self, name):
        cui = u""
        # check against names in the training data
        cui = Sieve.getTerminologyNameCui(Sieve.trainingDataTerminology.getNameToCuiListMap(), name)
        if cui != u"":
            return cui

        # check against names in the dictionary
        return Sieve.getTerminologyNameCui(Sieve.standardTerminology.getNameToCuiListMap(), name)

    @classmethod
    def getTrainingDataTerminology(self):
        return Sieve.trainingDataTerminology


class SimpleNameSieve(Sieve):

    @classmethod
    def getTerminologySimpleNames(self, phraseTokens):
        newPhrases = list()
        if len(phraseTokens) == 3 :
            newPhrase = phraseTokens[0]+" "+phraseTokens[2]
            newPhrases = Util.setList(newPhrases, newPhrase)
            newPhrase = phraseTokens[1]+" "+phraseTokens[2]
            newPhrases = Util.setList(newPhrases, newPhrase)

        return newPhrases



def makedir_and_clear(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)

def init(opt):
    # training_data_dir = opt.train
    # test_data_dir = opt.test
    # terminologyFile = opt.dict

    Ling.setStopwordsList(os.path.join(opt.resource, 'stopwords.txt'))
    Abbreviation.setWikiAbbreviationExpansionMap(os.path.join(opt.resource, 'ncbi-wiki-abbreviations.txt'))
    Ling.setDigitToWordformMapAndReverse(os.path.join(opt.resource, 'number.txt'))
    Ling.setSuffixMap(os.path.join(opt.resource, 'suffix.txt'))
    Ling.setPrefixMap(os.path.join(opt.resource, 'prefix.txt'))
    Ling.setAffixMap(os.path.join(opt.resource, 'affix.txt'))


    MultiPassSieveNormalizer.maxSieveLevel = opt.max_sieve

def runMultiPassSieve(opt):
    Sieve.setStandardTerminology(opt.dict)
    Sieve.setTrainingDataTerminology(opt.train)

    dataset = Documents.getDataSet(opt.test)
    for concepts in dataset:
        cuiNamesMap = dict()
        for concept in  concepts.getConcepts():
            MultiPassSieveNormalizer.applyMultiPassSieve(concept)
            if concept.getCui() == u"":
                concept.setCui(u"CUI-less")

            cuiNamesMap = Util.setMap(cuiNamesMap, concept.getCui(), concept.getName())

        AmbiguityResolution.start(concepts, cuiNamesMap)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', default='/Users/feili/project/disorder-normalizer/ncbi-data/training')
    parser.add_argument('-test', default='/Users/feili/project/disorder-normalizer/ncbi-data/test')
    parser.add_argument('-output', default='/Users/feili/project/disorder-normalizer/ncbi-data/output')
    parser.add_argument('-dict', default='/Users/feili/project/disorder-normalizer/ncbi-data/TERMINOLOGY.txt')
    parser.add_argument('-resource', default='/Users/feili/project/disorder-normalizer/resources')
    parser.add_argument('-max_sieve', type=int, default=1)

    opt = parser.parse_args()

    init(opt)



    makedir_and_clear(opt.output)

    runMultiPassSieve(opt)

    shutdownJVM()

    Evaluation.computeAccuracy()
    Evaluation.printResults()