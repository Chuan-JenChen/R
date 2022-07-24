# 有10 位學生學生參加數學、國文和英語考試，為了將學生進行比較，要將學生各個科目的成績進一步分析
# 要將前20%(含)的學生評為A，21~40%的學生評為B，以此類推。最後再依學生的姓名的字母進行排序

Student <- c("Sam Davis", "Kyle Anderson", "Jarrett Allen",
             "David Jones", "Bam Adebayo", "Cheryl Cushing", 
             "Udoka Azubuike", "Precious Achiuwa",  "Joel England", 
             "Mary Rayburn")
Math <- c(504, 701, 423, 368, 395, 617, 314, 523, 663, 632)
Chinese <- c(93, 99, 75, 83, 72, 65, 85, 65, 78, 76)
English <- c(22, 24, 17, 14, 23, 24, 29, 13, 24, 15)
df <- data.frame(Student, Math, Chinese, English, 
                     stringsAsFactors=FALSE)
df
str(df)

#請同學用各種學習過的函數觀察數據，可以發現一些問題。首先，三科考試的成績是無法比較的。因為它們的平均值和標準差相去甚遠，所以用平均值來衡量是沒有意義的。所以在衡量考試成績之前，必須將其變換為可比較的單元。
# 其次，為了給學生評等級，需要一種方法來確定某個學生在前述得分上百分比排名。
# 最後，表示姓名全部含在一個變數裡，為了正確地將其依姓和名排序，需要將姓和名拆開。
# 以上思考點，請各位同學依以下步驟進行拆解分析

# Step 1. 原始的學生冊如上，請限定輸出小數點後兩位的數字，讓輸出更容易閱讀。
options(digits = 2)

# Step 2. 由於數學、國文和英語考試的數值不同（均值和標準差相去甚遠），在組合之前需要先讓它們變得可以比較。請將變量進行標準化，這樣每科考試的成績就都是用單位標準差來表示，而不是以原始的尺度來表示。(hint:scale()函數)
Maths <- (scale(Math))
Maths
Chineses <- (scale(Chinese))
Chineses
Englishes <- (scale(English))
Englishes

#Step 3. 計算每人的所有科目的平均值，以獲得綜合得分，並將其添加到原始名冊中（hint: mean(), cbind())
averaged<- cbind (Maths, Chineses, Englishes)
averaged
averagedMean <- rowMeans(averaged)
averagedMean

#Step 4. 使用函數quantile()得到學生綜合得分的百分位數(0.8, 0.6, 0.4, 0.2)
quantile(averagedMean,c(0.8, 0.6, 0.4, 0.2))

df <- data.frame(Maths, Chineses, Englishes, averagedMean, 
                 stringsAsFactors=TRUE)
df


#Step 5. 將學生的排名，進行重編碼，為A,B,C,D,E，這新的五個類別型變量命名為grade。

"A"->df$grade [averagedMean >= 0.50]
"B"->df$grade [averagedMean < 0.50 & averagedMean >= 0.1 ] 
"C"->df$grade [averagedMean < 0.1 & averagedMean >= -0.3] 
"D"->df$grade [averagedMean < -0.3 & averagedMean >= -0.6]
"E"->df$grade [averagedMean < -0.6 ]
df

#Step 6. 以空格為界把學生姓名拆分為姓氏和名字，並返回一個name的列表(hint:使用strsplit())

StudentName <- strsplit(Student, ' ')
StudentName

#Step 7. 使用函數sapply()提取列表中每個成分的第一個元素，放入一個儲存名字的向量，並提取每個成分的第二個元素，放入一個儲存姓氏的向量。"["是一個可以提取某個對象的一部分的函數——在這裡它是用來提取列表name各成分中的第一個或第二個元素的。使用cbind()把它們添加到名冊中。由於已經不再需要student變量，請將其刪除。
FirstName <- sapply(StudentName, function(x) x[1])
FirstName
LastName <- sapply(StudentName, function(x) x[2])
LastName
#Step 8. 依姓氏(Lastname)和名字(Firstname)對數據集進行A-Z排序(hint:order()或arrange()函數)
Sort<-cbind (LastName,FirstName,df)
Sort

Sort[order(Sort$LastName,Sort$FirstName),]

