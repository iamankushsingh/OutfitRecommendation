const express = require('express');
const app = express();
app.get('/recommendations', (req, res) => {
    // Call the Python script to generate recommendations
    const pythonScript = spawn('python', ['recommendation_model.py']);
    pythonScript.stdout.on('data', (data) => {
      res.send(data.toString());
    });
  });
const multer = require('multer');
const upload = multer({ dest: './uploads/' });

app.post('/get-recommendations', upload.single('input_image'), (req, res) => {
    const inputImage = req.file.path;
    const outfitType = req.body.outfit_type;
    const color = req.body.color;

    const recommender = new FashionRecommender(dataset);
    const recommendations = recommender.get_recommendations(inputImage, outfitType, color);

    res.json(recommendations);
});

app.listen(3000, () => {
    console.log('Server started on port 3000');
});