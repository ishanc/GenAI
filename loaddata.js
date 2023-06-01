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

dotenv.config();

const API_KEY = process.env.OPENAI_API_KEY;
const RESOURCE_ENDPOINT = "https://wemopenai.openai.azure.com/";
const deployment_name = "text-embedding-ada-002";
const txtFilename = "BenInfo";
let question = "";
let answer = "";
const txtPath = "HealthTree.txt";//`./${txtFilename}.txt`;
const VECTOR_STORE_PATH = txtFilename+ '.index';
const CITATION_VECTOR_STORE_PATH = 'citations.index';
 question = "Where can i learn about 401k?";

openai.apiType = 'azure';
openai.apiKey = API_KEY;
openai.apiBase = RESOURCE_ENDPOINT;
openai.apiVersion = '2022-12-01';


const embed = new OpenAIEmbeddings({
    azureOpenAIApiVersion: "2023-03-15-preview",
    azureOpenAIApiKey: API_KEY,
    azureOpenAIApiInstanceName: "wemopenai",
    azureOpenAIApiDeploymentName: "text-embedding-ada-002",

});

async function callAPI() {

    let vectorStore;
    const text = fs.readFileSync(txtPath, 'utf8');
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 2000, chunkOverlap: 0 });
    const docs = await textSplitter.createDocuments([text]);
    if (fs.existsSync(VECTOR_STORE_PATH)) {
        console.log('Vector exists...');



        vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, embed);
        
//adds new file info to existing vectors
      for (let i = 0; i < docs.length; i++) {
            await vectorStore.addDocuments([docs[i]], embed);
            console.log(i);
        }

        console.log("loaded docs");

        await vectorStore.save(VECTOR_STORE_PATH);
      


        console.log("vector values: " + vectorStore);
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

    //sim search finnnnnally
    console.log("at sim serach" + question);
    const simSearch = await vectorStore.similaritySearch(question, 5);
    console.log('SimSearch : 1' + simSearch[0].pageContent + "simSearch 2: " + simSearch[0].pageContent);

    
    const citations = fs.readFileSync('citations.txt', 'utf8').split('\n');
    const texts = [];
    for (const citation of citations) {
        
        texts.push(citation);
    }
    console.log("texts split into array");
    const citdocs = await textSplitter.createDocuments(texts);
    console.log("loaded cit docs"+citdocs[0]);
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
    console.log(bestMatch);
    //check to see if there is a URL in there, if so output it
    const urlRegex = /(https?:\/\/\S+?)(?:\.\s|\.\)|\.)(?=\s|$)/;
    const urlMatch = bestMatch.match(urlRegex);
    let url1 = "";
    if (urlMatch) {
         url1 = urlMatch[1];
        console.log(url1);
    } else {
        console.log("No URL found in the text.");
    }


    let bestMatchURL = citations[bestMatch];
    console.log("Best matching URL: " + url1);


}
callAPI();

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







    