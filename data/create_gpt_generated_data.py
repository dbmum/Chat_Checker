import pandas as pd
from openai import OpenAI


def generate_article(title):
    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {
        "role": "system",
        "content": "You are the top journalist at NPR news organization. I will give you the title of an article that you need to write. Make each article at least 200 words."
      },
      {
        "role": "user",
        "content": title
      }
    ],
    temperature=1,
    max_tokens=300,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return ' '.join(response.choices[0].message.content.split()[:200]) #Number of words to return




def main():
    # Read the Parquet file into a Pandas DataFrame
    df = pd.read_parquet('data/articles.parquet')

    df['generated_content'] = df['Title'].apply(lambda x: generate_article(x))

    output_parquet = 'data/articles.parquet'
    df.to_parquet(output_parquet)

    print("Content generation complete")

if __name__ == '__main__':
    main()