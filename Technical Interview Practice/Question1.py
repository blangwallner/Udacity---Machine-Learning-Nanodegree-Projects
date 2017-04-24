# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
def question1(s,t):
    """
    Given two strings s and t, this function determines whether some
    anagram of t is a substring of s. Function returns True or False.
    For example: if s = "udacity" and t = "ad", then the function returns True. 
    """
    if s == '':
        perm_t_in_s = False
    elif t == '':
        perm_t_in_s = True
    else:
        from itertools import permutations
        perm_t = [''.join(p) for p in permutations(t)]
        perm_t_in_s = sum([s.count(tt) for tt in perm_t])>0
    return perm_t_in_s


def question1_a(s,t):
    """
    Given two strings s and t, this function determines whether some
    anagram of t is a substring of s. Function returns True or False.
    For example: if s = "udacity" and t = "ad", then the function returns True. 
    """
    def permutated_list(s):
        len_s = len(s)
        if len_s == 1:
            perm_list = [s]
        else:
            templist = permutated_list(s[0:-1])
            perm_list = [ t[0:j]+s[-1]+t[j:] for t in templist for j in range(0,len_s)]
        return perm_list
    
    if s == '':
        perm_t_in_s = False
    elif t == '':
        perm_t_in_s = True
    else:
        perm_t = permutated_list(t)
        perm_t_in_s = sum([s.count(tt) for tt in perm_t])>0
    return perm_t_in_s
    

s1 = "abczdefghiafwfweffafasdfasdfeasdfadferddfsdfwe"
s2 = "edzcbfhgi"
import time
t1 = time.time()
print question1(s1,s2)
delta1 = time.time() - t1
t1
print question1_a(s1,s2)
delta2 = time.time() - t1

print delta1
print delta2
