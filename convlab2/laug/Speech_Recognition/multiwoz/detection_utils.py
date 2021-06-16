import locale;
 
NUMBER_CONSTANT = {0:"zero ", 1:"one", 2:"two", 3:"three", 4:"four", 5:"five", 6:"six", 7:"seven",
                8:"eight", 9:"nine", 10:"ten", 11:"eleven", 12:"twelve", 13:"thirteen",
                14:"fourteen", 15:"fifteen", 16:"sixteen", 17:"seventeen", 18:"eighteen", 19:"nineteen" };
IN_HUNDRED_CONSTANT = {2:"twenty", 3:"thirty", 4:"forty", 5:"fifty", 6:"sixty", 7:"seventy", 8:"eighty", 9:"ninety"}
BASE_CONSTANT = {0:" ", 1:"hundred", 2:"thousand", 3:"million", 4:"billion"};

#supported number range is 1-n billion;
def translateNumberToEnglish(number):
    if str(number).isnumeric():
        if str(number)[0] == '0' and len(str(number)) > 1:
            return translateNumberToEnglish(int(number[1:]));
        if int(number) < 20:
            return NUMBER_CONSTANT[int(number)];
        elif int(number) < 100:
            if str(number)[1] == '0':
                return IN_HUNDRED_CONSTANT[int(str(number)[0])];
            else:
                return IN_HUNDRED_CONSTANT[int(str(number)[0])] + " " + NUMBER_CONSTANT[int(str(number)[1])];
        else:
            #locale.setlocale(locale.LC_ALL, "English_United States.1252");
            #strNumber = locale.format("%d"    , number, grouping=True);
            strNumber=str(number)
            numberArray = str(strNumber).split(",");
            stringResult = "";
            groupCount = len(numberArray) + 1;
            for groupNumber in numberArray:
                if groupCount > 1 and groupNumber[0:] != "000":
                    stringResult += str(getUnderThreeNumberString(str(groupNumber))) + " ";
                else:
                    break;
                groupCount -= 1;
                if groupCount > 1:
                    stringResult += BASE_CONSTANT[groupCount] + " ";
            endPoint = len(stringResult) - len(" hundred,");
            #return stringResult[0:endPoint];
            return stringResult;
                
    else:
        print("please input a number!");
 
#between 0-999
def getUnderThreeNumberString(number):
    if str(number).isnumeric() and len(number) < 4:
        if len(number) < 3:
            return translateNumberToEnglish(int(number));
        elif len(number) == 3 and number[0:] == "000":
            return " ";
        elif len(number) == 3 and number[1:] == "00":
            return NUMBER_CONSTANT[int(number[0])] + "  " + BASE_CONSTANT[1];
        else:    
            return NUMBER_CONSTANT[int(number[0])] + "  " + BASE_CONSTANT[1] + " and " + translateNumberToEnglish((number[1:]));
    
def translateTimeToEnglish(t):
    t=t.split(':')
    if t[1]!='00':
      return translateNumberToEnglish(t[0])+' '+translateNumberToEnglish(t[1])
    else:
      return translateNumberToEnglish(t[0])+' '+'o\'clock'

def span_typer(s):
    if s.isnumeric():
        return "number"
    if s.find(':')>=0:
        s=s.split(':')
        if len(s)==2:
            if s[0].isnumeric() and s[1].isnumeric():
                return "time"
    return "none"

def replacer(s):
    s=s.replace(' n\'t','n\'t')
    s=s.replace(' \'ll','\'ll')
    s=s.replace('centre','center')
    s=s.replace('-star',' star')
    s=s.replace('guesthouse','guest house')
    return s