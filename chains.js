const openai = require('openai');
const axios = require('axios');
const dotenv = require('dotenv');
const request = require("request");
const { OpenAIEmbeddings } = require("langchain/embeddings/openai");
const { HNSWLib } = require("langchain/vectorstores/hnswlib");
const fs = require("fs");
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const express = require('express');
const app = express();
const cors = require('cors');
const { RetrievalQAChain } = require('langchain/chains');
const { ChatOpenAI } = require('langchain/chat_models/openai');
const { BufferWindowMemory } = require('langchain/memory');
const { BufferMemory } = require('langchain/memory');
const { ConversationalRetrievalQAChain } = require('langchain/chains');
const { PromptTemplate } = require("langchain/prompts");
const { SystemChatMessage } = require("langchain/schema");
const { ChatPromptTemplate } = require("langchain/prompts");
const { VectorStoreToolkit, createVectorStoreAgent, initializeAgentExecutorWithOptions } = require("langchain/agents")
const { VectorDBQAChain } = require("langchain/chains");
const { ChainTool } = require("langchain/tools");

dotenv.config();



const https = require('https');// IC 
const http = require('http');//IC

const API_KEY = process.env.OPENAI_API_KEY;
const RESOURCE_ENDPOINT = "https://wemopenai.openai.azure.com/";
const deployment_name = "text-embedding-ada-002";
const txtFilename = "BenInfo";
const txtFilename1 = "rates";
let question = "";
let answer = "";
const txtPath = "extrainfo1.txt";//`./${txtFilename}.txt`;
const VECTOR_STORE_PATH = txtFilename+ '.index';
const CITATION_VECTOR_STORE_PATH = 'citations.index';
 question = "Where can i find info on my 401k?";

openai.apiType = 'azure';
openai.apiKey = API_KEY;
openai.apiBase = RESOURCE_ENDPOINT;
openai.apiVersion = '2022-12-01';


app.use(cors({
    origin: '*',
}));




const systemMessage = new SystemChatMessage("Can you respond to me like a pirate?");
 const ChatOAI = new ChatOpenAI({
    azureOpenAIApiVersion: "2023-03-15-preview",
    azureOpenAIApiKey: API_KEY,
    azureOpenAIApiInstanceName: "wemopenai",
    azureOpenAIApiDeploymentName: "gpt-35-turbo",
    temperature: 0,
    systemMessage: systemMessage,
});



const embed = new OpenAIEmbeddings({
    azureOpenAIApiVersion: "2023-03-15-preview",
    azureOpenAIApiKey: API_KEY,
    azureOpenAIApiInstanceName: "wemopenai",
    azureOpenAIApiDeploymentName: "text-embedding-ada-002",

});

//const memory = new FlatFileMemory();
const memory = new BufferMemory({
    memoryKey: "chat_history",
});
const conversational_memory = new BufferWindowMemory({
    memoryKey: "chat_history",
    returnMessage: true,
    k: 10
});


//if there is a vectorDB loaded then great use the tool if not 
let vectorStore;
let toolkit;
let chain;
let agent;
let tools;
let vectorStore_Info;
let vectorStoreTool;
let chain2;
let textSplitter;
async function callVector() {

    
    const text = fs.readFileSync(txtPath, 'utf8');
     textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 2000, chunkOverlap: 0 });
    const docs = await textSplitter.createDocuments([text]);

    if (fs.existsSync(VECTOR_STORE_PATH)) {
        console.log('Vector exists...');



         vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, embed);



    } else {
        //console.log(docs);
        vectorStore = await HNSWLib.fromDocuments([docs[0]], embed);

        for (let i = 1; i < docs.length; i++) {
            await vectorStore.addDocuments([docs[i]], embed);
            console.log(i);
        }

        console.log("loaded docs");

        await vectorStore.save(VECTOR_STORE_PATH);
    }
     chain = ConversationalRetrievalQAChain.fromLLM(ChatOAI, vectorStore.asRetriever(),
        { memory: conversational_memory }, { systemMessage: systemMessage });
    console.log({ VectorStoreToolkit });
     vectorStore_Info = ({
        name: "knowledge_base",
        description: "Be as detailed as possible in the answers, including observations that were used in final answer. ",
         vectorStore: vectorStore,
        memory: conversational_memory,
    });

     toolkit = new VectorStoreToolkit(vectorStore_Info, ChatOAI);

     chain2 = VectorDBQAChain.fromLLM(ChatOAI, vectorStore);



     vectorStoreTool = new ChainTool({
        name: "vector-store-qa",
        description: "You are helpful at provide exact answers and will provide all of the details in your response. If being asked about questions about RATES for plans you will always use this tool. You will not hallucinate or provide information outside of the data. If you cannot find information in the data, please state what information is missing and do not hallucinate informatoin. Here is context to help In addition you will use the conversation history to help answer follow up questions.",
        chain: chain2,
    })
    tools = [vectorStoreTool];
    console.log("Loaded agent.");

    console.log({ tools });
    agent = await initializeAgentExecutorWithOptions(tools, ChatOAI, {
        agentType: "chat-conversational-react-description",
        verbose: true,
    });
  

}



app.get('/users', (req, res) => {

    

    console.log(memory + "loaded mem");
  

    
    let question = "";
    if (!req.headers['question']) {

        question = "";
    } else {
        res.header('Access-Control-Allow-Origin', '*');
        question = req.headers['question'];

    }


    async function callAPI() {

      

        ////sim search finnnnnally
        if (vectorStore) {
            console.log("vector store is full");
        } else {
            await callVector();
        }
        console.log("vectore store length"+vectorStore.length);
        console.log("at sim serach" + question);
        const simSearch = await vectorStore.similaritySearch(question, 5);
        console.log('SimSearch : 1' + simSearch[0].pageContent + "simSearch 2: " + simSearch[1].pageContent);

        console.log(simSearch);
        const docIds = simSearch.map(result => result.id);
        const allVecotrs = simSearch;
        const filteredArray = await allVecotrs.filter(doc => docIds.includes(doc.id));
        const filteredStore = await HNSWLib.fromDocuments([filteredArray[0]], embed);
        console.log({ filteredStore });
        for (let i = 1; i < filteredArray.length; i++) {
            await filteredStore.addDocuments([filteredArray[i]], embed);
            console.log(i);
        }
        //let pSim = simSearch[0].pageContent.replace(/\n|\*|•/g, '');
        //console.log(pSim);
        //lets define a prompt template
        let template = "All information on this page is relevant to the question. Please do not make up things, use the data strictly. Please do not refer to the previous sentence in your answer. Question: {question} ?.";
        let prompt = new PromptTemplate({
            template: template,
            inputVariables: ["question", "context"]

        })

        //system message


        console.log(systemMessage);
        let tempRes = await prompt.format({ question: question, context: simSearch[0].pageContent });
        //console.log(tempRes);

        chain = ConversationalRetrievalQAChain.fromLLM(ChatOAI, vectorStore.asRetriever(),
            { memory: conversational_memory }, { systemMessage: systemMessage });
        console.log({ chain });

        const chatHistory = conversational_memory.chatHistory;

        //question = question +  JSON.stringify(chatHistory.messages.AIChatMessage) ;
        question = tempRes;
        //agent = createVectorStoreAgent(ChatOAI, toolkit, { memory: conversational_memory });
        //agent.verbose = true;
        //agent.memory = ({ conversational_memory });

        //const res1 = await chain.call({ question });
        const input = question;
        //const result = await agent.call({ input });
        const resp = await agent.call({ input });
        console.log(agent.memory.chatHistory.messages);
        



        //begin integrations of citations
        const citations = fs.readFileSync('citations.txt', 'utf8').split('\n');
        const texts = [];
        for (const citation of citations) {

            texts.push(citation);
        }
        console.log("texts split into array");
        const citdocs = await textSplitter.createDocuments(texts);
        console.log("loaded cit docs" + citdocs[0]);
        if (fs.existsSync(CITATION_VECTOR_STORE_PATH)) {
            vectorStore = await HNSWLib.load(CITATION_VECTOR_STORE_PATH, embed);
            console.log("Cit Vector Store Loaded");
        } else {
            vectorStore = await HNSWLib.fromDocuments([citdocs[0]], embed);

            for (let i = 1; i < citdocs.length; i++) {
                await vectorStore.addDocuments([citdocs[i]], embed);
                console.log(i);
            }

            console.log("loaded docs");

            await vectorStore.save(CITATION_VECTOR_STORE_PATH);
        }

        //query = "Where can i learn about 401k??";

        let simCitSearch = await vectorStore.similaritySearch(question, 5);
        let bestMatch = simCitSearch[0].pageContent;
        console.log(simCitSearch[0]);
        console.log(simCitSearch[1]);
        console.log(simCitSearch[2]);
        console.log(simCitSearch[3]);
        console.log(bestMatch);
        //check to see if there is a URL in there, if so output it
        const urlRegex = /URL:(.*?)\s\|/;
        const urlMatch = bestMatch.match(urlRegex);

        // Split the string into an array
        let nameArray = bestMatch.split('|');

        // The Page Name is the second element in the array. We remove the "Page Name:" part by splitting again and getting the second part.
        let pageName = nameArray[1].split(':')[1].trim();


        let url1 = "";
        if (urlMatch) {
            url1 = urlMatch[1];
            console.log(url1);
        } else {
            console.log("No URL found in the text.");
        }


        let bestMatchURL = citations[bestMatch];
        console.log("Best matching URL: " + url1);

        //end integration of citations







        res.json([
            {
                name: 'John Doe',
                email: 'johndoe@example.com',
                answer: resp.output,
                url: url1,
                pageName: pageName,

            },
            {
                name: 'Jane Doe',
                email: 'janedoe@example.com',
                prompt: "",
            },
        ]);

      


        //const citations = fs.readFileSync('citations.txt', 'utf8').split('\n');
        //const texts = [];
        //for (const citation of citations) {

        //    texts.push(citation);
        //}
        //console.log("texts split into array");
        //const citdocs = await textSplitter.createDocuments(texts);
        //console.log("loaded cit docs"+citdocs[0]);
        //if (fs.existsSync(CITATION_VECTOR_STORE_PATH)) {
        //    vectorStore = await HNSWLib.load(CITATION_VECTOR_STORE_PATH, embed);
        //    console.log("Cit Vector Store Loaded");
        //} else {
        //    vectorStore = await HNSWLib.fromDocuments([citdocs[0]], embed);

        //    for (let i = 1; i < citdocs.length; i++) {
        //        await vectorStore.addDocuments([citdocs[i]], embed);
        //        console.log(i);
        //    }

        //    console.log("loaded docs");

        //    await vectorStore.save(CITATION_VECTOR_STORE_PATH);
        //}

        ////query = "Where can i learn about 401k??";

        //let simCitSearch = await vectorStore.similaritySearch(question, 5);
        //let bestMatch = simCitSearch[0].pageContent;
        //console.log(bestMatch);
        ////check to see if there is a URL in there, if so output it
        //const urlRegex = /(https?:\/\/\S+?)(?:\.\s|\.\)|\.)(?=\s|$)/;
        //const urlMatch = bestMatch.match(urlRegex);
        //let url1 = "";
        //if (urlMatch) {
        //     url1 = urlMatch[1];
        //    console.log(url1);
        //} else {
        //    console.log("No URL found in the text.");
        //}


        //let bestMatchURL = citations[bestMatch];
        //console.log("Best matching URL: " + url1);


    }
    callAPI();

});


const httpsServer = https.createServer({
 // key: fs.readFileSync('C:/Certbot/live/www.thestore.co.in/privkey.pem'),
 // cert: fs.readFileSync('C:/Certbot/live/www.thestore.co.in/fullchain.pem'),
 key: fs.readFileSync('C:/Users/Administrator/Downloads/SSL-20230629T171214Z-001/SSL/generated-private-key.pem'),
 cert: fs.readFileSync('C:/Users/Administrator/Downloads/SSL-20230629T171214Z-001/SSL/5ca3dfe8dce791b0.crt'),
ca: fs.readFileSync('C:/Users/Administrator/Downloads/SSL-20230629T171214Z-001/SSL/gd_bundle-g2-g1.crt'),
}, app);

httpsServer.listen(443, () => {
    console.log('HTTPS Server running on port 443');
});

//app.get('/users', (req, res) => {

//    async function callAPI() {

    
//    // Get all of the users from the database.
//    const question = req.headers['question'];
//    const contextPrompt = req.headers['contextprompt'];
//    let prompt = contextPrompt;
//    let answer = "";
    
//    answer = await runWithEmbeddings(question);
//    console.log("this is answer: " + answer);
//    if (contextPrompt) {
//        prompt = "completed something with the request";
//    }
//    console.log(req.headers);
//    res.json([
//        {
//            name: 'John Doe',
//            email: 'johndoe@example.com',
//            answer: answer,
//        },
//        {
//            name: 'Jane Doe',
//            email: 'janedoe@example.com',
//            prompt: prompt,
//        },
//    ]);

//    }//end async
//    callAPI();
//});//end of app


//app.listen(3000);











//embeddings

//console.log(HNSWLib);

//console.log("here"+vectorStore);



//const url = `https://wemopenai.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-03-15-preview`;


//const jsonData = {
//    "input": "The food was delicious and the waiter...",
//    "input": "what is the food..."
//};




//const headers = {
//    "Content-Type": "application/json",
//    "api-key": API_KEY,
//};

//const body = {
//    jsonData
//};


//request({
//    url,
//    headers,
//    body: jsonData,
//    method: "POST",
//    json: true,
//}, (err, response, body) => {
//    if (err) {
//        console.log(err);
//    } else {
       
        
        
//        //if (fs.existsSync(VECTOR_STORE_PATH)) {
//        //    console.log('Vector exists...');
//        //} else {

//        //    //load vectors
//        //    const embeddingsLoad = [
//        //        [1, 2, 3],
//        //        [4, 5, 6],
//        //        [7, 8, 9],
//        //    ];

//        //    const embeddings = [];
//        //    const space = 'cosine'; // Replace 3 with the actual dimensionality of your vectors

//        //    // ...

//        //    const args = {
//        //        space: space.toString(), // Convert the number of dimensions to a string
//        //        numDimensions: body.data[0].embedding.length, // Convert the number of dimensions to a string
//        //    };

//        //    const vectorStore = new HNSWLib(body.data, args);
//        //    vectorStore.addVectors(body.data, body.data);
//        //    console.log(vectorStore);

//        //    if (embeddings) {
//        //        console.log("The embeddings were successfully added to the vecotre store.");
//        //    } else {
//        //        console.log("the embeddings were not successfully added to the vectorstore.");
//        //    }
//        //    console.log(VECTOR_STORE_PATH);
//        //    //vectorStore.save(VECTOR_STORE_PATH);
//        //    console.log(vectorStore);
//        //}
        
//    }
//});







    