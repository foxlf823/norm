
class FoxTokenizer:
    white_char = set()
    # white char
    white_char.add(' ')
    white_char.add('\f')
    white_char.add('\n')
    white_char.add('\r')
    white_char.add('\t')
    white_char.add('\v')

    # punctuation begin
    punc = set()
    punc.add('`')
    punc.add('~')
    punc.add('!')
    punc.add('@')
    punc.add('#')
    punc.add('$')
    punc.add('%')
    punc.add('&')
    punc.add('*')
    punc.add('(')
    punc.add(')')
    punc.add('-')
    punc.add('_')
    punc.add('+')
    punc.add('=')
    punc.add('{')
    punc.add('}')
    punc.add('|')
    punc.add('[')
    punc.add(']')
    punc.add('\\')
    punc.add(':')
    punc.add(';')
    punc.add('\'')
    punc.add('"')
    punc.add('<')
    punc.add('>')
    punc.add(',')
    punc.add('.')
    punc.add('?')
    punc.add('/')
    # punctuation end
        
        

    # given a 'offset' and a string 's'
    # return a list of tokens, each element is a list (text, start, end)
    @classmethod
    def tokenize(self, offset, s, onlyText):
        sb = ''
        tokens = []
        i = 0

        while i<len(s):
            ch = s[i]

            if ch in self.white_char or ch in self.punc:
                if len(sb) != 0:

                    if onlyText == False:
                        token = []
                        token.append(sb)
                        token.append(offset-len(sb))
                        token.append(offset)
                        tokens.append(token)
                    else:
                        tokens.append(sb)
                    sb = ''

                if ch not in self.white_char:

                    if onlyText == False:
                        token = []
                        token.append(ch)
                        token.append(offset)
                        token.append(offset+1)
                        tokens.append(token)
                    else:
                        tokens.append(ch)
            else:
                sb += ch

            offset += 1
            i += 1

        if len(sb) != 0:

            if onlyText == False:
                token = []
                token.append(sb)
                token.append(offset-len(sb))
                token.append(offset)
                tokens.append(token)
            else:
                tokens.append(sb)
            sb = ''

        return tokens


if __name__ == '__main__':



    tokens = FoxTokenizer.tokenize(0, "hello,  world", True)

    print(tokens)

    with open("token_example.txt") as fp:
        text = fp.read()
        tokens = FoxTokenizer.tokenize(0, text, False)
        print(tokens)

