import re
PUNCTUATION = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

URL_PATTERN = re.compile(r"""
                         @?\s*? # optional leading @ 
                         https?\s*:\s*// # http:// or https://
                         \s*[\w-]+ # domain 
                         (?:\s*\.\s*[\w-]+)+ # dot separated domain parts (e.g. .com)
                         (?:\s*/\s*[\w\-./%~]*)? # optional path, include characters / . - % ~ 
                         (?:\s*\.\s*\w+)? # optional extension (e.g. .html)
                         """,
                         re.IGNORECASE | re.VERBOSE)

MAIL_PATTERN = re.compile(r"""\b[\w\.]+ # part before @
                           @
                           \w+(?:\s*\.\s*\w+)+\b # part after @ with spaces allowed
                           """,
                          re.IGNORECASE | re.VERBOSE)


PHONE_PATTERN = re.compile(r"""
                            (?<!\w)              # not preceded by a word char
                            (?:\+?\d{1,3}[\s\-\.]?)?  # optional country code, like +33, +1, 0044
                            (?:\(?\d{2,4}\)?[\s\-\.]?) # area code, with or without ()
                            (?:\d[\s\-\.]?){6,10}      # remaining digits
                            (?!\w)               # not followed by a word char
                            """)
