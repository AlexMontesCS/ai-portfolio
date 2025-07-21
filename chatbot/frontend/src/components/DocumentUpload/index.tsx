import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, Trash2, Download, Eye } from 'lucide-react';
import { format } from 'date-fns';
import { Document } from '../../types';
import { Button, Card, LoadingSpinner, Alert } from '../ui';
import { useDocuments, useUploadDocument, useDeleteDocument } from '../../hooks/useApi';

export const DocumentUpload: React.FC = () => {
  const { data: documentsData, isLoading, error } = useDocuments();
  const uploadMutation = useUploadDocument();
  const deleteMutation = useDeleteDocument();
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    for (const file of acceptedFiles) {
      try {
        setUploadProgress(prev => ({ ...prev, [file.name]: 0 }));
        await uploadMutation.mutateAsync(file);
        setUploadProgress(prev => ({ ...prev, [file.name]: 100 }));
        
        // Clear progress after a delay
        setTimeout(() => {
          setUploadProgress(prev => {
            const newProgress = { ...prev };
            delete newProgress[file.name];
            return newProgress;
          });
        }, 2000);
      } catch (error) {
        setUploadProgress(prev => {
          const newProgress = { ...prev };
          delete newProgress[file.name];
          return newProgress;
        });
        console.error('Upload failed:', error);
      }
    }
  }, [uploadMutation]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/markdown': ['.md', '.markdown'],
      'text/plain': ['.txt'],
    },
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  const handleDelete = async (documentId: string) => {
    if (window.confirm('Are you sure you want to delete this document?')) {
      try {
        await deleteMutation.mutateAsync(documentId);
      } catch (error) {
        console.error('Delete failed:', error);
      }
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getDocumentTypeIcon = (type: string) => {
    switch (type) {
      case 'pdf':
        return '📄';
      case 'markdown':
        return '📝';
      case 'text':
        return '📄';
      default:
        return '📄';
    }
  };

  if (error) {
    return (
      <Alert type="error" title="Error loading documents">
        Failed to load documents. Please try again.
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <Card title="Upload Documents">
        <>
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive
                ? 'border-blue-400 bg-blue-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            <input {...getInputProps()} />
            <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            {isDragActive ? (
              <p className="text-blue-600 font-medium">Drop the files here...</p>
            ) : (
              <div>
                <p className="text-gray-600 font-medium mb-2">
                  Drag & drop files here, or click to select
                </p>
                <p className="text-sm text-gray-500">
                  Supports PDF, Markdown, and text files (max 10MB)
                </p>
              </div>
            )}
          </div>

          {/* Upload Progress */}
          {Object.keys(uploadProgress).length > 0 ? (
            <div className="mt-4 space-y-2">
              {Object.entries(uploadProgress).map(([filename, progress]: [string, number]) => (
                <div key={filename} className="flex items-center space-x-3">
                  <LoadingSpinner size="sm" />
                  <span className="text-sm text-gray-600">{filename}</span>
                  <div className="flex-1 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <span className="text-sm text-gray-500">{progress}%</span>
                </div>
              ))}
            </div>
          ) : null}

          {uploadMutation.error && (
            <Alert type="error" className="mt-4">
              {(() => {
                const error = uploadMutation.error;
                if (error instanceof Error) {
                  return `Upload failed: ${error.message}`;
                }
                return 'Upload failed. Please try again.';
              })()}
            </Alert>
          )}
        </>
      </Card>

      {/* Documents List */}
      <Card
        title="Uploaded Documents"
        actions={
          <span className="text-sm text-gray-500">
            {documentsData?.total || 0} documents
          </span>
        }
      >
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <LoadingSpinner size="lg" />
            <span className="ml-3 text-gray-600">Loading documents...</span>
          </div>
        ) : documentsData?.documents.length === 0 ? (
          <div className="text-center py-8">
            <FileText className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              No documents uploaded
            </h3>
            <p className="text-gray-500">
              Upload your first document to get started with the chatbot.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {documentsData?.documents.map((document) => (
              <DocumentItem
                key={document.id}
                document={document}
                onDelete={handleDelete}
                isDeleting={deleteMutation.isLoading}
              />
            ))}
          </div>
        )}
      </Card>
    </div>
  );
};

interface DocumentItemProps {
  document: Document;
  onDelete: (id: string) => void;
  isDeleting: boolean;
}

const DocumentItem: React.FC<DocumentItemProps> = ({
  document,
  onDelete,
  isDeleting,
}) => {
  return (
    <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50">
      <div className="flex items-center space-x-3 flex-1 min-w-0">
        <span className="text-2xl">
          {getDocumentTypeIcon(document.document_type)}
        </span>
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-medium text-gray-900 truncate">
            {document.filename}
          </h4>
          <div className="flex items-center space-x-4 text-sm text-gray-500">
            <span>{formatFileSize(document.size)}</span>
            <span>{document.chunk_count} chunks</span>
            <span>
              {format(new Date(document.upload_time), 'MMM d, yyyy')}
            </span>
          </div>
        </div>
      </div>
      
      <div className="flex items-center space-x-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onDelete(document.id)}
          disabled={isDeleting}
        >
          {isDeleting ? (
            <LoadingSpinner size="sm" />
          ) : (
            <Trash2 className="w-4 h-4" />
          )}
        </Button>
      </div>
    </div>
  );
};

function getDocumentTypeIcon(type: string): string {
  switch (type) {
    case 'pdf':
      return '📄';
    case 'markdown':
      return '📝';
    case 'text':
      return '📄';
    default:
      return '📄';
  }
}

function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
