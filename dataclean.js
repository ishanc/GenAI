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
const pdfs = require('pdf-parse');


dotenv.config();

const API_KEY = process.env.OPENAI_API_KEY;
const RESOURCE_ENDPOINT = "https://wemopenai.openai.azure.com/";
const deployment_name = "text-embedding-ada-002";
const txtFilename = "BenInfo";
let question = "";
let answer = "";
const txtPath = "extrainfo1.txt";//`./${txtFilename}.txt`;
const VECTOR_STORE_PATH = txtFilename+ '.index';
const CITATION_VECTOR_STORE_PATH = 'citations.index';
const pdfLocation = 'Rates-2023.pdf';
let textArray = [];

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
let page;
async function splitPDF(inputFilePath) {
    const dataBuffer = fs.readFileSync(inputFilePath);
    

    try {
        const pdfData = await pdfs(dataBuffer);
        const totalPages = pdfData.numpages;
        const pdfText = pdfData.text;

        // Split the PDF text based on page numbers

        console.log(totalPages);
        for (let pageNumber = 1; pageNumber <= totalPages; pageNumber++) {
            const pageSeparator = '23AE-Standard-Medical-Rates';//'\n' + pageNumber;
            //console.log(pageSeparator);
            const pageTexts = pdfText.split(pageSeparator);
            
            textArray.push(pageTexts[pageNumber]);
           
        }

         // Output the array of page texts
    } catch (error) {
        console.error(error);
    }
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 50000, chunkOverlap: 32 });
    const pdfDocc = await textSplitter.createDocuments(textArray);
    

    if (fs.existsSync('rates.index')) {
        console.log('Vector exists...');



        vectorStore = await HNSWLib.load('rates.index', embed);


        console.log("loaded docs");

        for (let i = 0; i < pdfDocc.length; i++) {
            await vectorStore.addDocuments([pdfDocc[i]], embed);
            console.log(i);
        }

        console.log("loaded docs after");

        await vectorStore.save('rates.index');


        console.log("vector values: " + vectorStore);
    } else {
        //console.log(docs);
        vectorStore = await HNSWLib.fromDocuments([pdfDocc[0]], embed);

        for (let i = 1; i < pdfDocc.length; i++) {
            await vectorStore.addDocuments([pdfDocc[i]], embed);
            console.log(i);
        }

        console.log("loaded docs");

        await vectorStore.save('rates.index');
    }




}
//splitPDF(pdfLocation);

async function callAPI() {

    let vectorStore;
    const cleanText = fs.readFileSync(txtPath, 'utf8').split('\n\n');;
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 50000, chunkOverlap: 32 });

    const info = [];
    for (const clean of cleanText) {

        info.push(clean);
    }

    //load pdf

    
    const docs = await textSplitter.createDocuments(info);
    if (fs.existsSync(VECTOR_STORE_PATH)) {
        console.log('Vector exists...');



        vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, embed);
        

        console.log("loaded docs");

        for (let i = 0; i < docs.length; i++) {
            await vectorStore.addDocuments([docs[i]], embed);
            console.log(i);
        }

        console.log("loaded docs after");

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









    