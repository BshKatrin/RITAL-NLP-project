import re

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

PUNCTUATION = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
