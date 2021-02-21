import torch

class MergeModel(torch.nn.Module):
    '''
            Todo
    '''
    def __init__(self, summary_model, sentiment_model):
        super().__init__()
        self.summary_model = summary_model
        self.sentiment_model = sentiment_model
    
    def forward(self, article_id, article_mask, summary_id, summary_mask):
        self.summary_out = self.summary_model(article_id,  article_mask, summary_id, summary_mask)

        idx =  torch.argmax(self.summary_out.logits, dim=2, keepdims=  True)
        mask = torch.zeros_like(self.summary_out.logits).scatter_(2, idx, 1.)
        
        pred_summary_id = (self.summary_out.logits * mask).sum(axis=2).long()
        pred_summary_mask = torch.ones_like(summary_mask).long()
        pred_summary_id[:, -1] = 2
        
        self.article_sentiment = self.sentiment_model(article_id,  article_mask, labels=None)
        self.summary_sentiment = self.sentiment_model(pred_summary_id,  pred_summary_mask, labels=None)
        
        return [self.summary_out, self.article_sentiment, self.summary_sentiment]