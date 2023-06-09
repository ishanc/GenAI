FROM node:19-alpine
COPY package.json /app/
COPY GenAI /app/
WORKDIR /app
RUN npm install
CMD ["node","app.js"]