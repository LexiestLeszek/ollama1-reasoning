import ollama
import json
import time

def llm_call(messages, max_tokens, is_final_answer=False):
    for attempt in range(3):
        try:
            response = ollama.chat(
                model='llama3.1:8b-instruct-q5_K_M',
                messages=messages,
                options={"temperature":0.2, "max_length":max_tokens},
                format='json',
            )
            return json.loads(response['message']['content'])
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {"title": "Error", "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}
                else:
                    return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}", "next_action": "final_answer"}
            time.sleep(1)  # Wait for 1 second before retrying

def generate_response(prompt):
    messages = [
        {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES."""},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        start_time = time.time()
        step_data = llm_call(messages, 300)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))
        
        messages.append({"role": "assistant", "content": json.dumps(step_data)})
        
        if step_data['next_action'] == 'final_answer' or step_count > 25:  # Maximum of 25 steps to prevent infinite thinking time
            break
        
        step_count += 1

    # Generate final answer
    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})
    
    start_time = time.time()
    final_data = llm_call(messages, 200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    steps.append(("Final Answer", final_data['content'], thinking_time))

    return steps, total_thinking_time

def main():
    print("Ollama1: Local reasoning like o1 ")
    
    user_query = input("Enter your query (e.g., How many 'R's are in the word strawberry?): ")
    
    if user_query:
        print("Generating response...")
        steps, total_thinking_time = generate_response(user_query)
        
        for title, content, thinking_time in steps:
            print(f"\n{title}")
            print("-" * len(title))
            print(content)
            print(f"Thinking time: {thinking_time:.2f} seconds")
        
        print(f"\nTotal thinking time: {total_thinking_time:.2f} seconds")

if __name__ == "__main__":
    main()
