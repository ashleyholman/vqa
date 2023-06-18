import React, { useState, useEffect } from 'react';
import { Link, useLocation, useParams, useRoutes } from 'react-router-dom';
import './ErrorAnalysisTable.css';
import AnswerClassSampleCategoryViewer from './AnswerClassSampleCategoryViewer.js'

function AnswerClassSampleViewer({ runId, errorAnalysisSummaryData}) {
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
  }, [runId]);

  useEffect(() => {
    const pathParts = location.pathname.split('/').filter(Boolean);

    console.log("pathParts: ", pathParts)

    if (pathParts.length > 4) {
      const fifthPart = pathParts[4];

      switch(fifthPart) {
        case 'tp':
          setTab('tp');
          break;
        case 'fp':
          setTab('fp');
          break;
        case 'fn':
          setTab('fn');
          break;
        case 'tn':
          setTab('tn');
          break;
        default:
          setTab('tp');
      }
    } else {
      setTab('tp');
    }
  }, [location]);

  let tp_data = null;
  let fp_data = null;
  let fn_data = null;

  if (sampleQuestionData) {
    tp_data = sampleQuestionData['TP']['sample_questions'];
    fp_data = sampleQuestionData['FP']['sample_questions'];
    fn_data = sampleQuestionData['FN']['sample_questions'];
  }

  let element = useRoutes([
    { path: '', element: <AnswerClassSampleCategoryViewer title='True Positives' sampleQuestions={tp_data} errorAnalysisSummaryData={errorAnalysisSummaryData}/> },
    { path: 'tp', element: <AnswerClassSampleCategoryViewer title='True Positives' sampleQuestions={tp_data} errorAnalysisSummaryData={errorAnalysisSummaryData}/> },
    { path: 'fp', element: <AnswerClassSampleCategoryViewer title='False Positives' sampleQuestions={fp_data} errorAnalysisSummaryData={errorAnalysisSummaryData}/> },
    { path: 'fn', element: <AnswerClassSampleCategoryViewer title='False Negatives' sampleQuestions={fn_data} errorAnalysisSummaryData={errorAnalysisSummaryData}/> }
  ]);

  if (!sampleQuestionData) {
    return <div>Loading...</div>;
  }

  return (
    <div class="answer-class-viewer">
      <div class="answer-class-viewer-header">
        <div className="tab-bar">
          <Link to={`/run/${runId}/error_analysis/${classId}/tp`} className={tab === 'tp' ? 'active-tab' : ''} onClick={(e) => {setTab('tp')}}>True Positives ({`${tp_data.length}`})</Link>
          <Link to={`/run/${runId}/error_analysis/${classId}/fp`} className={tab === 'fp' ? 'active-tab' : ''} onClick={(e) => {setTab('fp')}}>False Positives ({`${fp_data.length}`})</Link>
          <Link to={`/run/${runId}/error_analysis/${classId}/fn`} className={tab === 'fn' ? 'active-tab' : ''} onClick={(e) => {setTab('fn')}}>False Negatives ({`${fn_data.length}`})</Link>
        </div>
      </div>
      {element}
    </div>
  );
}

export default AnswerClassSampleViewer;