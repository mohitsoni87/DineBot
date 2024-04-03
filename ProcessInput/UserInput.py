from langchain_core.prompts import PromptTemplate

from langchain_openai import OpenAI
from langchain.chains import LLMChain
import json, re
from SemanticAnalysis.SemanticAnalysis import analyseUserInput

llm = OpenAI(api_key="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", temperature=0.7)

restTemplate = '''Reset Earlier Output JSON and freshly Give the output in strictly one JSON object in one line and nothing else:
extract rest_type, location, and cuisine from Users input and fill the variables with the values else strictly give ""
Example of Output format (everything in small) in JSON: "rest_type": "","cuisine": "" "location":"" with curly brackets
User input is {var}
Precisely and very cautiously check for the values of the variables and then only give the values else mostly give "".
Populate location only if you find noun or exact name for a location. Note: Friendly place, staff or live music is not a location, it is mostly one worded. I repeat no other characters outside the JSON object'''

restprompt = PromptTemplate(
    input_variables=['var'],
    template=restTemplate)

restLLMChain = LLMChain(llm=llm, prompt=restprompt)

convTemplate = '''Reply in True or False. is the input a Greeting Input or the user is trying to ask something in terms of a restaurant? True if Hi, Hello kind else False -- {var}'''
convPrompt = PromptTemplate(
    input_variables=['var'],
    template=convTemplate)
convLLMChain = LLMChain(llm=llm, prompt=convPrompt)

casualTemplate = '''Act as a Restaurant Recommender Chat bot and reply to - {var}'''
casualPrompt = PromptTemplate(
    input_variables=['var'],
    template=casualTemplate)
casualLLMChain = LLMChain(llm=llm, prompt=casualPrompt)


semanticResponseTemplate = '''Act as a restaurant Recommender bot and Transform data into readable format for the user with appropriate labels such name, location, price for two and contact number - {var}. Each restaurant will have one row (either in numbering or bulletin) and separate each attribute with a comma. Always start the answer as if you're recommending for e.g. Here are some choices which you may like. Dont give any extra characters in the beginning'''
semanticResponsePrompt = PromptTemplate(
    input_variables=['var'],
    template=semanticResponseTemplate)
semanticResponseLLMChain = LLMChain(llm=llm, prompt=semanticResponsePrompt)

def processUserInput(userInput):
    print("UserInput: ", userInput)
    try:
        isCasualConv = convLLMChain(userInput)

        if "true" in isCasualConv['text'].lower():
            casualReply = casualLLMChain.run(userInput)
            return casualReply

        response = restLLMChain.run(userInput)
        print("Initial Response", response)

        pattern = r'\{.*?\}'

        # Find all matches in the input string
        matches = re.findall(pattern, response)
        print("RegEx match", matches)

        if len(matches) > 0:
            responseObject = eval(str(matches[0]))
            # Check if any attribute is null
            is_null = True
            for value in responseObject.values():
                # print(value, "FOR")
                if (value == "null" or value == "" or value == None):
                    pass
                else:
                    is_null = False
                    break
            if (is_null):
                print("Trigger Engine 2..")
                response = analyseUserInput(userInput)
                llmSemanticResponse = semanticResponseLLMChain.run(response)
                print("Semantic Output is\n", llmSemanticResponse)
                regex_pattern = r'Here are some.*$'

                # Use re.search() to find the first match
                match = re.search(regex_pattern, llmSemanticResponse, re.DOTALL)

                if match:
                    matched_string = match.group()
                    return matched_string
                return "Unfortunately the Engine was not able to find any Restaurant similar to your input. Would you like to try something else?"
            else:
                return "Trigger Engine 1.."
        else:
            print("Retry")
    except Exception as e:
        print(e, True)
        return "I couldn't understand you properly. Could you please try again"


def processUserConversationInput(userInput):
    response = convLLMChain.run(userInput)
    print(response)
    return response
