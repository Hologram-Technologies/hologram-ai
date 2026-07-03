fetch("https://huggingface.co/api/models/meta-llama/Llama-3.1-8B", {credentials: "include"}).then(res => console.log(res.headers)).catch(err => console.error(err));
