import React, { useState } from 'react';
import './AnswerClassSampleCategoryViewer.css';

function AnswerClassSampleCategoryViewer({ title, sampleQuestions, errorAnalysisSummaryData }) {
  const [currentIndex, setCurrentIndex] = useState(0);

  if (sampleQuestions.length === 0) {
    return <div>No data</div>;
  }

  const currentSampleQuestion = sampleQuestions[currentIndex];

  const handleNext = () => {
    setCurrentIndex((currentIndex + 1) % sampleQuestions.length);
  }

  const handlePrev = () => {
    setCurrentIndex((currentIndex - 1 + sampleQuestions.length) % sampleQuestions.length);
  }

  const formatImageId = (id) => {
    return String(id).padStart(12, '0');
  }

  const imageUrl = `${process.env.REACT_APP_COCO_IMAGE_HOST}/static/media/vqa-validation-images/COCO_val2014_${formatImageId(currentSampleQuestion.image_id)}.jpg`

  return (
    <div>
      <h2>{currentSampleQuestion.question_text}</h2>
      <img src={imageUrl} alt="vqa_image" />
      <h3>Predictions</h3>
      <ol>
        {currentSampleQuestion.predicted_classes.map((predicted_class, index) => (
          <li key={index} className={`${index === 0 ? 'first-answer' : ''} ${predicted_class.class_id === currentSampleQuestion.true_class ? 'correct-answer' : 'wrong-answer'}`}>
            {errorAnalysisSummaryData[predicted_class.class_id].class_label} ({(predicted_class.confidence * 100).toFixed(2)}%)
          </li>
        ))}
      </ol>
      <h3>Correct answer: {errorAnalysisSummaryData[currentSampleQuestion.true_class].class_label}</h3>
      <button onClick={handlePrev}>Previous</button>
      <button onClick={handleNext}>Next</button>
    </div>
  );
}

export default AnswerClassSampleCategoryViewer;