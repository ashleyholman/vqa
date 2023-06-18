import React, { useState, useEffect } from 'react';
import { Link, useLocation, useParams, useRoutes } from 'react-router-dom';
import AnswerClassSampleCategoryViewer from './AnswerClassSampleCategoryViewer.js'

function AnswerClassSampleViewer({ runId }) {
  const [sampleQuestionData, setSampleQuestionData] = useState(null);
  const { classId } = useParams();
  const [tab, setTab] = useState('tp');
  const location = useLocation();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`/data/run/${runId}/error_analysis/class_${classId}.json`);
        const data = await response.json();
        setSampleQuestionData(data);
      } catch (error) {
        console.error('Error:', error);
      }
    };

    fetchData();
  }, [runId, classId]);

  useEffect(() => {
    const pathParts = location.pathname.split('/').filter(Boolean);
    const fifthPart = pathParts.length > 4 ? pathParts[4] : 'tp';
    setTab(['tp', 'fp', 'fn', 'tn'].includes(fifthPart) ? fifthPart : 'tp');
  }, [location]);

  const getData = (category) => sampleQuestionData && sampleQuestionData[category.toUpperCase()]['sample_questions'];

  const categories = {
    'tp' : "True Positives",
    'fp' : "False Positives",
    'fn' : "False Negatives"
  }

  const elements = useRoutes([
    { path: '', element: <AnswerClassSampleCategoryViewer categoryType='tp' sampleQuestions={getData('tp')} runId={runId} classId={classId}/> },
    ...Object.keys(categories).map((category) => ({
      path: `${category}/*`,
      element: <AnswerClassSampleCategoryViewer categoryType={category} sampleQuestions={getData(category)} runId={runId} classId={classId}/>
    }))
  ]);

  if (!sampleQuestionData) {
    return <div>Loading...</div>;
  }

  return (
    <div className="answer-class-viewer">
      <div className="answer-class-viewer-header">
        <div className="tab-bar">
          {Object.entries(categories).map(([category, description]) => (
            <Link
              key={category}
              to={`/run/${runId}/error_analysis/${classId}/${category}`}
              className={tab === category ? 'active-tab' : ''}
              onClick={() => setTab(category)}>
              {`${description}`} ({getData(category)?.length})
            </Link>
          ))}
        </div>
      </div>
      {elements}
    </div>
  );
}

export default AnswerClassSampleViewer;