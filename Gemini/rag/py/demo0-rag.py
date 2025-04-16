from dotenv import dotenv_values
from google import genai
from google.genai import types
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
import chromadb

# Define a helper to retry when per-minute quota is reached.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

secrets = dotenv_values(".env")
GOOGLE_API_KEY = secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=GOOGLE_API_KEY)

class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            ),
        )
        return [e.values for e in response.embeddings]

def main():
   print('Gemini SDK genai version ->', genai.__version__)

   for m in client.models.list():
    if "embedContent" in m.supported_actions:
        print(m.name)
        
   DOCUMENT1 = "Operating the Climate Control System  Your Googlecar has a climate control system that allows you to adjust the temperature and airflow in the car. To operate the climate control system, use the buttons and knobs located on the center console.  Temperature: The temperature knob controls the temperature inside the car. Turn the knob clockwise to increase the temperature or counterclockwise to decrease the temperature. Airflow: The airflow knob controls the amount of airflow inside the car. Turn the knob clockwise to increase the airflow or counterclockwise to decrease the airflow. Fan speed: The fan speed knob controls the speed of the fan. Turn the knob clockwise to increase the fan speed or counterclockwise to decrease the fan speed. Mode: The mode button allows you to select the desired mode. The available modes are: Auto: The car will automatically adjust the temperature and airflow to maintain a comfortable level. Cool: The car will blow cool air into the car. Heat: The car will blow warm air into the car. Defrost: The car will blow warm air onto the windshield to defrost it."
   DOCUMENT2 = 'Your Googlecar has a large touchscreen display that provides access to a variety of features, including navigation, entertainment, and climate control. To use the touchscreen display, simply touch the desired icon.  For example, you can touch the "Navigation" icon to get directions to your destination or touch the "Music" icon to play your favorite songs.'
   DOCUMENT3 = "Shifting Gears Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position.  Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions."

   documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]
   
   DB_NAME = "googlecardb"

   embed_fn = GeminiEmbeddingFunction()
   embed_fn.document_mode = True
   
   try:
      chroma_client = chromadb.Client()
      # Here comes the llm embedding function. In this case defined as a class.
      # You can also use the default embedding function, which is a simple embedding function that returns the input as the output.
      db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)
   
      db.add(documents=documents, ids=[str(i) for i in range(len(documents))])
   
      db.count()
      # You can peek at the data too.
      #db.peek(1)
      # Query the database
      query = "How do you use the touchscreen to play music?"

      result = db.query(query_texts=[query], n_results=1)
      [all_passages] = result["documents"]
      
      print("all ->", all_passages[0])
      
      # Switch to query mode when generating embeddings.
      embed_fn.document_mode = False
      # I will also switch the model down below to gemini flash
      
      #Augmented generation: Answer the question
      query_oneline = query.replace("\n", " ")
      
      # This prompt is where you can specify any guidance on tone, or what topics the model should stick to, or avoid.
      prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below. 
      Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
      However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
      strike a friendly and converstional tone. If the passage is irrelevant to the answer, you may ignore it.
      
      QUESTION: {query_oneline}
      """
      
      # Add the retrieved documents to the prompt.
      for passage in all_passages:
          passage_oneline = passage.replace("\n", " ")
          prompt += f"PASSAGE: {passage_oneline}\n"
      
      print('final prompt -> ', prompt)

      answer = client.models.generate_content(
      model="gemini-2.0-flash",
      contents=prompt)

      print('Gemini answer ->', answer.text)
            
      
   finally:
      # Clean up
      chroma_client.delete_collection(DB_NAME)
      # chroma_client.close() #neither chromadb nor genai need to be closed
      # Close the client
      # client.close()
      print("Done.")

if __name__ == "__main__":
   main()
   
   