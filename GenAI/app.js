//const openai = require('openai');
const fs = require('fs');

const path = require('path');
const configFilePath = path.join(__dirname, 'config.json');
const configDetails = JSON.parse(fs.readFileSync(configFilePath));

// Setting up the deployment name
const chatgptModelName = configDetails.CHATGPT_MODEL;

// Set the API key for your OpenAI resource
const openaiAPIKey = process.env.OPENAI_API_KEY;

// Set the base URL for your OpenAI resource
const openaiAPIBase = configDetails.OPENAI_API_BASE;

// Set the OpenAI API version
const openaiAPIVersion = configDetails.OPENAI_API_VERSION;

// Create an instance of the OpenAI API
const openaiApi = new openai.LanguageCompletionApi({
    apiKey: openaiAPIKey,
    apiBase: openaiAPIBase,
    apiVersion: openaiAPIVersion
});

// Define the input message
const messages = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Who won the world series in 2020?' }
];

// Define the API call
const completionParams = {
    engine: chatgptModelName,
    messages: messages
};

// Call the OpenAI API
openaiApi.createCompletion(completionParams)
    .then(response => {
        // Print the generated response
        console.log(response.choices[0].message.content);
    })
    .catch(error => {
        // Handle API errors
        if (error instanceof openai.errors.APIError) {
            console.error(`OpenAI API returned an API Error: ${error}`);
        } else if (error instanceof openai.errors.AuthenticationError) {
            console.error(`OpenAI API returned an Authentication Error: ${error}`);
        } else if (error instanceof openai.errors.APIConnectionError) {
            console.error(`Failed to connect to OpenAI API: ${error}`);
        } else if (error instanceof openai.errors.InvalidRequestError) {
            console.error(`Invalid Request Error: ${error}`);
        } else if (error instanceof openai.errors.RateLimitError) {
            console.error(`OpenAI API request exceeded rate limit: ${error}`);
        } else if (error instanceof openai.errors.ServiceUnavailableError) {
            console.error(`Service Unavailable: ${error}`);
        } else if (error instanceof openai.errors.Timeout) {
            console.error(`Request timed out: ${error}`);
        } else {
            console.error('An exception has occurred.');
        }
    });