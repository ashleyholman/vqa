import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import './AnswerClassSampleCategoryViewer.css';

function AnswerClassSampleCategoryViewer({ runId, classId, categoryType, sampleQuestions, errorAnalysisSummaryData }) {
  const location = useLocation();
  const pathParts = location.pathname.split('/').filter(Boolean);
  const currentIndex = pathParts.length > 5 ? Number(pathParts[5]) : 0;

  const nextIndex = (currentIndex + 1) % sampleQuestions.length;
  const prevIndex = (currentIndex - 1 + sampleQuestions.length) % sampleQuestions.length;

  if (sampleQuestions.length === 0) {
    return <div>No data</div>;
  }

  const currentSampleQuestion = sampleQuestions[currentIndex];

  const formatImageId = (id) => {
    return String(id).padStart(12, '0');
  }

  const imageUrl = `${process.env.REACT_APP_COCO_IMAGE_HOST}/static/media/vqa-validation-images/COCO_val2014_${formatImageId(currentSampleQuestion.image_id)}.jpg`

  return (
    <div>
      <div id="prev-next-links">
        <Link to={`/run/${runId}/error_analysis/${classId}/${categoryType}/${prevIndex}`}>{' << '} Previous</Link>{' '}
        <span>{currentIndex + 1} / {sampleQuestions.length}</span>{' '}
        <Link to={`/run/${runId}/error_analysis/${classId}/${categoryType}/${nextIndex}`}>Next {' >> '}</Link>
      </div>
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
    </div>
  );
}

export default AnswerClassSampleCategoryViewer;