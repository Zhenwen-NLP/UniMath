import torch
import torchvision
from modeling_t5 import VLT5
from vqvae import VQVAE

def build_model():
    cnn = getattr(torchvision.models, 'resnet101')(pretrained=True)
    layers = [cnn.conv1,
              cnn.bn1,
              cnn.relu,
              cnn.maxpool]
    for i in range(4):
        name = 'layer%d' % (i + 1)
        layers.append(getattr(cnn, name))
    model = torch.nn.Sequential(*layers)
    model.cuda()
    model.eval()
    return model


class VLT5Geo(VLT5):
    def __init__(self, config):
        super().__init__(config)
        self.image_encoder = VQVAE(hidden_channels = 1024, nb_levels=2, scaling_rates=[8,4],beta = 0.25)

    def train_step(self, batch):

        device = next(self.parameters()).device
        mwp_indexs = [i for i, x in enumerate(batch['problem_form']) if x == 'mwp']
        #not_mwp_indexs = [i for i, x in enumerate(batch['problem_form']) if x != 'mwp']
        image = batch['image_list'].to(device)
        self.image_encoder = self.image_encoder.to(device)
        y, d, enc_out, _, _ = self.image_encoder(image)
        rloss = y.sub(image).pow(2)
        rloss[mwp_indexs,:,:,:] == 0
        #print('lloss',d.shape)
        r_loss, l_loss = rloss.mean(), sum(d)
        loss_vae = r_loss + self.image_encoder.beta * l_loss
        

        


        vis_feats = enc_out[-1]
        help_tensor = torch.ones_like(vis_feats).to(device)
        help_tensor[mwp_indexs,:,:,:] = 0
        vis_feats = vis_feats * help_tensor
        #print('feats',vis_feats.shape)
        N, C, H, W = vis_feats.shape
        vis_feats = vis_feats.reshape(N, C, -1).permute(0, 2, 1)

        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        #vis_pos[mwp_indexs,:,:] = 0
        vis_attention_mask = batch['vis_attention_mask'].to(device)

        lm_labels = batch["target_ids"].to(device)

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            vis_attention_mask=vis_attention_mask,
            labels=lm_labels,
            reduce_loss=True,
            return_dict=True
        )

        loss = output['loss']

        result = {
            'loss': loss,
            'loss_vae': loss_vae
        }
        return result

    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device
        image = batch['image_list'].to(device)
        mwp_indexs = [i for i, x in enumerate(batch['problem_form']) if x == 'mwp']
        self.image_encoder = self.image_encoder.to(device)

        y, d, enc_out, _, _ = self.image_encoder(image)
        vis_feats = enc_out[-1]
        help_tensor = torch.ones_like(vis_feats).to(device)
        help_tensor[mwp_indexs,:,:,:] = 0
        vis_feats = vis_feats * help_tensor

        N, C, H, W = vis_feats.shape
        vis_feats = vis_feats.reshape(N, C, -1).permute(0, 2, 1)

        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        vis_attention_mask = batch['vis_attention_mask'].to(device)

        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            vis_attention_mask=vis_attention_mask,
            **kwargs
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        result = {}
        result['pred'] = generated_sents

        return result
