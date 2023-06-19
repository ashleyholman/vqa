import React from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import { ThreeDots } from 'react-loader-spinner';
import ModalImage from 'react-modal-image';

import './AnswerClassSampleCategoryViewer.css';
import { ErrorAnalysisContext } from '../contexts/ErrorAnalysisContext.js';

function AnswerClassSampleCategoryViewer({ runId, classId, categoryType, sampleQuestions }) {
  const { errorAnalysisSummaryData, isErrorAnalysisSummaryDataLoaded } = React.useContext(ErrorAnalysisContext);
  const [isImageLoading, setImageLoading] = React.useState(true);
  const location = useLocation();
  const pathParts = location.pathname.split('/').filter(Boolean);
  const currentIndex = pathParts.length > 5 ? Number(pathParts[5]) : 0;
  const navigate = useNavigate();

  const nextIndex = (currentIndex + 1) % sampleQuestions.length;
  const prevIndex = (currentIndex - 1 + sampleQuestions.length) % sampleQuestions.length;

  const currentSampleQuestion = sampleQuestions.length > 0 ? sampleQuestions[currentIndex] : null

  React.useEffect(() => {
    setImageLoading(true);
  }, [currentSampleQuestion]);

  const handleImageLoad = () => {
    setImageLoading(false);
  };

    // Add useEffect hook for keyboard navigation
    React.useEffect(() => {
      function handleKeyDown(e) {
          switch (e.key) {
              case 'ArrowRight':
                  navigate(`/run/${runId}/error_analysis/${classId}/${categoryType}/${nextIndex}`);
                  break;
              case 'ArrowLeft':
                  navigate(`/run/${runId}/error_analysis/${classId}/${categoryType}/${prevIndex}`);
                  break;
              default:
                  break;
          }
      }
      window.addEventListener('keydown', handleKeyDown);

      return () => {
          window.removeEventListener('keydown', handleKeyDown);
      }
  }, [navigate, runId, classId, categoryType, nextIndex, prevIndex]);

  const formatImageId = (id) => {
    return String(id).padStart(12, '0');
  }

  const imageUrl = `${process.env.REACT_APP_COCO_IMAGE_HOST}/val2014/COCO_val2014_${formatImageId(currentSampleQuestion.image_id)}.jpg`

  React.useEffect(() => {
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => setImageLoading(false);
    img.onerror = () => setImageLoading(false);
  }, [imageUrl]);

  React.useEffect(() => {
    setImageLoading(true);
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => setImageLoading(false);
    img.onerror = () => setImageLoading(false);
  }, [imageUrl]);

  if (sampleQuestions.length === 0) {
    return <div>No data</div>;
  }

  if (!isErrorAnalysisSummaryDataLoaded) {
    return <div>Loading...</div>;
  }

  return (
    <div>
      <div id="prev-next-links">
        <Link to={`/run/${runId}/error_analysis/${classId}/${categoryType}/${prevIndex}`}>{' << '} Previous</Link>{' '}
        <span>{currentIndex + 1} / {sampleQuestions.length}</span>{' '}
        <Link to={`/run/${runId}/error_analysis/${classId}/${categoryType}/${nextIndex}`}>Next {' >> '}</Link>
      </div>
      <h2>{currentSampleQuestion.question_text}</h2>
      {isImageLoading &&
        <div className="image-loading">
          <ThreeDots color="#ffffff" height={50} width={50} />
        </div>
      }
      <div className={`image-content ${isImageLoading ? 'hidden' : ''}`}>
        <ModalImage
            small={imageUrl}
            large={imageUrl}
            alt="vqa_image"
            hideDownload={true}
            hideZoom={true}
          />
      </div>
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