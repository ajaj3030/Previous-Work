import right0 from '../../../assets/baby-animation/right/right-0.png';
import right1 from '../../../assets/baby-animation/right/right-1.png';
import right2 from '../../../assets/baby-animation/right/right-2.png';
import right3 from '../../../assets/baby-animation/right/right-3.png';

import left0 from '../../../assets/baby-animation/left/left-0.png';
import left1 from '../../../assets/baby-animation/left/left-1.png';
import left2 from '../../../assets/baby-animation/left/left-2.png';
import left3 from '../../../assets/baby-animation/left/left-3.png';

import up0 from '../../../assets/baby-animation/up/up-0.png';
import up1 from '../../../assets/baby-animation/up/up-1.png';
import up2 from '../../../assets/baby-animation/up/up-2.png';
import up3 from '../../../assets/baby-animation/up/up-3.png';

import down0 from '../../../assets/baby-animation/down/down-0.png';
import down1 from '../../../assets/baby-animation/down/down-1.png';
import down2 from '../../../assets/baby-animation/down/down-2.png';
import down3 from '../../../assets/baby-animation/down/down-3.png';

import v_down0 from '../../../assets/baby-animation/vodafone/down/down-0.png';
import v_down1 from '../../../assets/baby-animation/vodafone/down/down-1.png';
import v_down2 from '../../../assets/baby-animation/vodafone/down/down-2.png';
import v_down3 from '../../../assets/baby-animation/vodafone/down/down-3.png';

export const BasicAvatar = {
    id: "BASIC",
    directions: ['UP', 'DOWN', 'LEFT', 'RIGHT'],
    frames: {
      'RIGHT': [right0, right1, right2, right3],
      'LEFT': [left0, left1, left2, left3],
      'UP': [up0, up1, up2, up3],
      'DOWN': [down0, down1, down2, down3]
    },
    opposites: {
      'UP': 'DOWN',
      'DOWN': 'UP',
      'LEFT': 'RIGHT',
      'RIGHT': 'LEFT'
    }
}

export const VodafoneAvatar = {
  id: "BASIC",
  directions: ['UP', 'DOWN', 'LEFT', 'RIGHT'],
  frames: {
    'RIGHT': [right0, right1, right2, right3],
    'LEFT': [left0, left1, left2, left3],
    'UP': [up0, up1, up2, up3],
    'DOWN': [v_down0, v_down1, v_down2, v_down3]
  },
  opposites: {
    'UP': 'DOWN',
    'DOWN': 'UP',
    'LEFT': 'RIGHT',
    'RIGHT': 'LEFT'
  }
}

export const AllAvatars = {
    'BASIC':{
        id: "BASIC",
        directions: ['UP', 'DOWN', 'LEFT', 'RIGHT'],
        frames: {
        'RIGHT': [right0, right1, right2, right3],
        'LEFT': [left0, left1, left2, left3],
        'UP': [up0, up1, up2, up3],
        'DOWN': [down0, down1, down2, down3]
        },
        opposites: {
        'UP': 'DOWN',
        'DOWN': 'UP',
        'LEFT': 'RIGHT',
        'RIGHT': 'LEFT'
        }
    },
    'VODAFONE':{
      id: "BASIC",
      directions: ['UP', 'DOWN', 'LEFT', 'RIGHT'],
      frames: {
      'RIGHT': [right0, right1, right2, right3],
      'LEFT': [left0, left1, left2, left3],
      'UP': [up0, up1, up2, up3],
      'DOWN': [v_down0, v_down1, v_down2, v_down3]
      },
      opposites: {
      'UP': 'DOWN',
      'DOWN': 'UP',
      'LEFT': 'RIGHT',
      'RIGHT': 'LEFT'
      }
  }
}