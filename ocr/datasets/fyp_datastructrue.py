


#this is the structure of the specific question
{
    "subject": "Computer Science",
    "pastpaper_name": "COMP3230",
    "pastpaper_year": "2023",
    "question": "一个问题：例如：线程的定义",
    # "answer": "1 s",  这个可以先不要，因为有些pp我们找不到答案
    "question_type": "Multiple choice question",
    "position":"1",   #选择题第一题的意思
    "key_point": ["thread","defination"], #这个再议，先加上，里面的词汇以syllabus官方词汇为主,没有在syllabus中体现的话就自己编一下
    "marks": "3",
    "difficulty": "2", #1-10
    "time_require": "1", #单位 minutes 
    "repeatability": "yes",   #之前出现过类似的题目，如果是输入的最早年份pp，默认为no
    "repeatability_level": "high", # low, mid, high，no  代表这题重复的态度，如果是high的话，
                                   #希望下一年还尽量保持high,如果之前没出过，上一条为no的，本行也是no

}


#针对这个paper有一个总参数
{
    "subject": "Computer Science",
    "pastpaper_name": "COMP3230",
    "pastpaper_year": "2023",
    "totalmark":100,
    "difficulty_level_distribution": {"high":20,"mid":40,"low":40},
    "creative":{"repeatability":70,"new":30},
    "question_type":{"Multiple choice question":[10,#题目个数
                                                 3,#分数/题
                                                 7,#1-10 题型中内容每年重复率
                                                 3,#1-10,3+7=10，题目变化率
                                                 ],
                     "Long question":[]#相似
                     },
    "totaltime":180 ,#minutes
    
    
}