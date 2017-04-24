def question2(a):
    """
    Given a string a, function finds the longest palindromic substring contained in a.
    If there a several palindrome of the same length, the first one appearing in the
    given string a is returned
    """
    def is_palindrome(s):
        """
        Given a string s, function check whether s is a palindrome.  
        """
        len_s = len(s)
        # take first half of string; if length of string is odd then exclude middle character
        half1 = s[0:len_s//2]
        # take second half of string in reverse order; if length of string is odd then exclude middle character
        half2 = s[:(len_s+1)//2-1:-1]
        # check if first half and reversed second half coincide
        # note: if length of string is odd, then middle character is unimportant
        return half1 == half2

    len_a = len(a)
    if len_a == 0:
        # if a is empty, then so is the longest palindrome in it
        palin = ''
    elif len_a == 1:
        # if a is a single character, it is a palindrome itself
        palin = a
    else:
        # iterate over decreasing substring length
        str_len = len_a
        while 1:
            # note this loop will terminate since single characters are palindromes
            # Generate a list of all length str_len sub-strings of a
            sub_str = [a[i:i+str_len] for i in range(0,len_a-str_len+1)]
            # check if there are palindromes contained in the list sub_str
            palin_idx = [idx for idx, s in enumerate(sub_str) if is_palindrome(s)]
            # if there are palindromes, then take the first one appearing
            if palin_idx:
                palin = sub_str[palin_idx[0]]
                break
            # reduce substring length by one
            str_len -= 1
    return palin



a = 'abcdceabaczcabba'
print question2(a)

a = ''
print question2(a)

a = 'a'
print question2(a)